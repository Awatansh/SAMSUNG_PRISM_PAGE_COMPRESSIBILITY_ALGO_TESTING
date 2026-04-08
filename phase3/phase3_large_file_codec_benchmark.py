"""
Phase 3: Large-file compression benchmark

Compares the Phase 3 predictor against always-compress baselines on large files
such as images, audio, and video. The benchmark reports:

* Phase 3 selective page compression using the predictor
* Page-level baseline compression with common industry codecs
* Whole-file baseline compression with the same codecs
* Availability checks for optional codecs

The predictor is page-based, so file contents are chunked into 4 KB pages.
For the last partial page, zero-padding is used to preserve page semantics.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import importlib.util
import lzma
import os
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import lz4.frame
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required dependency 'lz4'. Install project dependencies with 'pip install -r requirements.txt'."
    ) from exc

try:
    from phase3.phase3_two_stage_predictor import two_stage_predict  # noqa: E402        
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Unable to import the Phase 3 predictor. Install project dependencies with 'pip install -r requirements.txt'."
    ) from exc


PAGE_SIZE = 4096

MEDIA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v",
}


@dataclass
class CodecSpec:
    name: str
    compress: Callable[[bytes], bytes]
    available: bool = True
    note: str = ""


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def build_codec_registry() -> list[CodecSpec]:
    codecs = [
        CodecSpec("lz4", lambda data: lz4.frame.compress(data, compression_level=16), True, "fast baseline used by the project"),
        CodecSpec("gzip", lambda data: gzip.compress(data, compresslevel=9), True, "deflate family"),
        CodecSpec("zlib", lambda data: zlib.compress(data, level=9), True, "deflate family"),
        CodecSpec("bz2", lambda data: bz2.compress(data, compresslevel=9), True, "bzip2"),
        CodecSpec("lzma", lambda data: lzma.compress(data, preset=6, format=lzma.FORMAT_XZ), True, "xz/lzma"),
    ]

    if has_module("brotli"):
        import brotli  # type: ignore

        codecs.append(CodecSpec("brotli", lambda data: brotli.compress(data, quality=11), True, "optional"))

    if has_module("zstandard"):
        import zstandard as zstd  # type: ignore

        zstd_compressor = zstd.ZstdCompressor(level=19)
        codecs.append(CodecSpec("zstd", lambda data: zstd_compressor.compress(data), True, "optional"))

    return codecs


def discover_files(
    roots: Iterable[Path],
    extensions: set[str] | None,
    min_size_bytes: int,
    recursive: bool,
) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            if (extensions is None or root.suffix.lower() in extensions) and root.stat().st_size >= min_size_bytes:
                files.append(root)
            continue

        if not root.exists():
            continue

        iterator = root.rglob("*") if recursive else root.glob("*")
        for path in iterator:
            if not path.is_file():
                continue
            if extensions is not None and path.suffix.lower() not in extensions:
                continue
            if path.stat().st_size < min_size_bytes:
                continue
            files.append(path)

    unique_files = sorted({path.resolve() for path in files})
    return unique_files


def chunk_to_pages(data: bytes) -> list[bytes]:
    if not data:
        return []

    pages: list[bytes] = []
    for offset in range(0, len(data), PAGE_SIZE):
        page = data[offset:offset + PAGE_SIZE]
        if len(page) < PAGE_SIZE:
            page = page + b"\x00" * (PAGE_SIZE - len(page))
        pages.append(page)
    return pages


def benchmark_codec_pages(pages: list[bytes], codec: CodecSpec) -> tuple[float, int]:    
    start = time.perf_counter()
    total_size = 0
    for page in pages:
        total_size += len(codec.compress(page))
    elapsed = time.perf_counter() - start
    return elapsed, total_size


def benchmark_codec_file(data: bytes, codec: CodecSpec) -> tuple[float, int]:
    start = time.perf_counter()
    compressed = codec.compress(data)
    elapsed = time.perf_counter() - start
    return elapsed, len(compressed)


def benchmark_phase3_selective(pages: list[bytes], codec: CodecSpec, seed: int | None = None) -> dict[str, object]:
    if seed is not None:
        np.random.seed(seed)

    predictor_time = 0.0
    compression_time = 0.0
    output_size = 0
    stage_counts: Counter[str] = Counter()

    for page in pages:
        t0 = time.perf_counter()
        decision, stage, _, _, _ = two_stage_predict(page)
        predictor_time += time.perf_counter() - t0
        stage_counts[stage] += 1

        if decision == "compressible":
            t1 = time.perf_counter()
            output_size += len(codec.compress(page))
            compression_time += time.perf_counter() - t1
        else:
            output_size += len(page)

    total_time = predictor_time + compression_time
    return {
        "predictor_time_s": predictor_time,
        "compression_time_s": compression_time,
        "total_time_s": total_time,
        "output_size_bytes": output_size,
        "stage_counts": dict(stage_counts),
    }


def format_rate(bytes_count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return bytes_count / seconds / (1024 * 1024)


def summarize_stage_counts(stage_counts: dict[str, int], total_pages: int) -> str:       
    parts = []
    for stage in ("text_detection", "stage1", "stage2"):
        count = stage_counts.get(stage, 0)
        parts.append(f"{stage}={count} ({count / total_pages * 100:.1f}%)")
    return ", ".join(parts)


def build_file_report(file_path: Path, codecs: list[CodecSpec], predictor_codec: CodecSpec, seed: int | None) -> dict[str, object]:
    data = file_path.read_bytes()
    pages = chunk_to_pages(data)

    if not pages:
        return {
            "file": str(file_path),
            "size_bytes": len(data),
            "pages": 0,
            "note": "empty file",
        }

    baseline_page_results = {}
    baseline_file_results = {}

    for codec in codecs:
        page_time, page_size = benchmark_codec_pages(pages, codec)
        file_time, file_size = benchmark_codec_file(data, codec)
        baseline_page_results[codec.name] = {
            "time_s": page_time,
            "output_size_bytes": page_size,
            "throughput_mb_s": format_rate(len(data), page_time),
        }
        baseline_file_results[codec.name] = {
            "time_s": file_time,
            "output_size_bytes": file_size,
            "throughput_mb_s": format_rate(len(data), file_time),
        }

    phase3 = benchmark_phase3_selective(pages, predictor_codec, seed=seed)
    phase3_throughput = format_rate(len(data), float(phase3["total_time_s"]))
    lz4_page_baseline = baseline_page_results.get("lz4")
    speedup_vs_lz4_page = None
    if lz4_page_baseline and phase3["total_time_s"] > 0:
        speedup_vs_lz4_page = lz4_page_baseline["time_s"] / float(phase3["total_time_s"])

    return {
        "file": str(file_path),
        "name": file_path.name,
        "size_bytes": len(data),
        "pages": len(pages),
        "phase3": phase3,
        "phase3_throughput_mb_s": phase3_throughput,
        "speedup_vs_lz4_page": speedup_vs_lz4_page,
        "baseline_page": baseline_page_results,
        "baseline_file": baseline_file_results,
    }


def print_available_codecs(codecs: list[CodecSpec]) -> None:
    print("Available codecs:")
    for codec in codecs:
        status = "available" if codec.available else "unavailable"
        extra = f" - {codec.note}" if codec.note else ""
        print(f"  {codec.name:8s}: {status}{extra}")


def print_file_report(report: dict[str, object], codecs: list[CodecSpec], predictor_codec: CodecSpec) -> None:
    print("\n" + "=" * 78)
    print(f"FILE: {report['file']}")
    print("=" * 78)
    print(f"Size: {report['size_bytes'] / (1024 * 1024):.2f} MiB")
    print(f"Pages: {report['pages']}")
    print()

    phase3 = report.get("phase3")
    if isinstance(phase3, dict):
        print(f"Phase 3 predictor-assisted compression ({predictor_codec.name} final codec):")
        print(f"  Predictor time:     {phase3['predictor_time_s'] * 1000:.2f} ms")       
        print(f"  Compression time:   {phase3['compression_time_s'] * 1000:.2f} ms")     
        print(f"  Total time:         {phase3['total_time_s'] * 1000:.2f} ms")
        print(f"  Throughput:         {report['phase3_throughput_mb_s']:.2f} MiB/s")     
        print(f"  Output size:        {phase3['output_size_bytes'] / (1024 * 1024):.2f} MiB")
        print(f"  Stage usage:        {summarize_stage_counts(phase3['stage_counts'], report['pages'])}")
        if report.get("speedup_vs_lz4_page") is not None:
            print(f"  Speedup vs lz4 page baseline: {report['speedup_vs_lz4_page']:.2f}x")
        print()

    baseline_page = report.get("baseline_page", {})
    baseline_file = report.get("baseline_file", {})

    print("Page-level always-compress baseline:")
    for codec in codecs:
        result = baseline_page.get(codec.name)
        if not result:
            continue
        print(
            f"  {codec.name:8s}: {result['time_s'] * 1000:8.2f} ms | "
            f"{result['throughput_mb_s']:8.2f} MiB/s | "
            f"output {result['output_size_bytes'] / (1024 * 1024):8.2f} MiB"
        )

    print()
    print("Whole-file always-compress baseline:")
    for codec in codecs:
        result = baseline_file.get(codec.name)
        if not result:
            continue
        print(
            f"  {codec.name:8s}: {result['time_s'] * 1000:8.2f} ms | "
            f"{result['throughput_mb_s']:8.2f} MiB/s | "
            f"output {result['output_size_bytes'] / (1024 * 1024):8.2f} MiB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Phase 3 predictor speed against common compression codecs on large files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--include-all-large",
        action="store_true",
        help="Include any large file, not just media extensions.",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=1.0,
        help="Minimum file size to include when scanning directories.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive directory scanning.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum number of files to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible predictor sampling.",
    )
    parser.add_argument(
        "--predictor-codec",
        choices=["lz4", "gzip", "zlib", "bz2", "lzma", "brotli", "zstd"],
        default="lz4",
        help="Codec used when the predictor decides a page should be compressed.",       
    )
    parser.add_argument(
        "--output-csv",
        default="phase3_large_file_benchmark_results.csv",
        help="CSV file for detailed per-file results.",
    )
    parser.add_argument(
        "--output-md",
        default="phase3_large_file_benchmark_summary.md",
        help="Markdown summary report.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify codec and input availability without running the benchmark.",  
    )
    args = parser.parse_args()

    codecs = [codec for codec in build_codec_registry() if codec.available]
    codec_map = {codec.name: codec for codec in codecs}
    predictor_codec = codec_map.get(args.predictor_codec)
    if predictor_codec is None:
        raise SystemExit(f"Predictor codec '{args.predictor_codec}' is not available in this environment.")

    print("=" * 78)
    print("PHASE 3 LARGE FILE COMPRESSION BENCHMARK")
    print("=" * 78)
    print()
    print("What this script checks:")
    print("  1. Whether the selected input files are large media-like files")
    print("  2. Whether Phase 3 can run on them with page-level prediction")
    print("  3. How Phase 3 compares against always-compress baselines")
    print("  4. How common compression standards compare on the same data")
    print()
    print_available_codecs(codecs)
    print()

    min_size_bytes = int(args.min_size_mb * 1024 * 1024)
    roots = [Path(path) for path in args.paths]
    extensions = None if args.include_all_large else set(MEDIA_EXTENSIONS)
    files = discover_files(
        roots,
        extensions=extensions,
        min_size_bytes=min_size_bytes,
        recursive=not args.no_recursive,
    )

    if not files and not args.include_all_large:
        print("No media files were found in the selected paths.")
        print("Use --include-all-large to benchmark any large files, or point the script at a media directory.")
        print("This means the benchmark is possible, but this workspace currently does not contain suitable inputs.")
        return

    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print("No files matched the selected filters.")
        return

    print(f"Benchmarking {len(files)} file(s) using page size {PAGE_SIZE} bytes.")       
    print(f"Predictor codec: {predictor_codec.name}")
    print(f"Minimum file size: {args.min_size_mb:.2f} MiB")
    print()

    if args.verify_only:
        print("Verification only requested.")
        print("Inputs are available and the codec registry loaded successfully.")        
        return

    reports: list[dict[str, object]] = []
    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path}")
        report = build_file_report(file_path, codecs, predictor_codec, seed=args.seed)   
        reports.append(report)
        print_file_report(report, codecs, predictor_codec)

    rows = []
    for report in reports:
        base = {
            "file": report.get("file"),
            "name": report.get("name", Path(str(report.get("file", ""))).name),
            "size_bytes": report.get("size_bytes"),
            "pages": report.get("pages"),
        }

        phase3 = report.get("phase3")
        if isinstance(phase3, dict):
            base.update({
                "phase3_predictor_time_s": phase3["predictor_time_s"],
                "phase3_compression_time_s": phase3["compression_time_s"],
                "phase3_total_time_s": phase3["total_time_s"],
                "phase3_output_size_bytes": phase3["output_size_bytes"],
                "phase3_stage_counts": repr(phase3["stage_counts"]),
                "phase3_throughput_mb_s": report["phase3_throughput_mb_s"],
                "speedup_vs_lz4_page": report.get("speedup_vs_lz4_page"),
            })

        for codec in codecs:
            page_result = report.get("baseline_page", {}).get(codec.name, {})
            file_result = report.get("baseline_file", {}).get(codec.name, {})
            base[f"page_{codec.name}_time_s"] = page_result.get("time_s")
            base[f"page_{codec.name}_output_size_bytes"] = page_result.get("output_size_bytes")
            base[f"page_{codec.name}_throughput_mb_s"] = page_result.get("throughput_mb_s")
            base[f"file_{codec.name}_time_s"] = file_result.get("time_s")
            base[f"file_{codec.name}_output_size_bytes"] = file_result.get("output_size_bytes")
            base[f"file_{codec.name}_throughput_mb_s"] = file_result.get("throughput_mb_s")

        rows.append(base)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_csv, index=False)

    summary_path = Path(args.output_md)
    summary_lines = [
        "# Phase 3 Large File Compression Benchmark",
        "",
        f"Files benchmarked: {len(files)}",
        f"Page size: {PAGE_SIZE} bytes",
        f"Predictor codec: {predictor_codec.name}",
        "",
        "## Codec availability",
    ]
    for codec in codecs:
        summary_lines.append(f"- {codec.name}: available{(' - ' + codec.note) if codec.note else ''}")
    summary_lines.extend([
        "",
        "## Notes",
        "- Phase 3 is page-based, so large files are evaluated in 4 KB chunks.",
        "- The predictor skips compression for pages judged incompressible.",
        "- The benchmark includes page-level and whole-file baselines for comparison.",  
        f"- Detailed results are saved in {args.output_csv}.",
    ])
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print("BATCH SUMMARY")
    print("=" * 78)
    print(results_df[["file", "size_bytes", "pages", "phase3_total_time_s", "phase3_throughput_mb_s", "speedup_vs_lz4_page"]].to_string(index=False))
    print()
    print(f"Saved detailed results to {args.output_csv}")
    print(f"Saved summary to {args.output_md}")
    print("The benchmark is possible in this workspace, but no media files were present by default.")


if __name__ == "__main__":
    main()