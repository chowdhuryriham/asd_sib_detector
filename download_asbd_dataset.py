#!/usr/bin/env python3
"""Build the ASBD data folder from the dataset workbook.

The workbook is a small XLSX file, so this script parses it with the Python
standard library instead of requiring openpyxl. Video downloading uses yt-dlp
when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile
import xml.etree.ElementTree as ET


XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DEFAULT_WORKBOOK = "ASBD - Autism Stimming Behavior Datase.xlsx"
DEFAULT_OUT_DIR = Path("data") / "asbd"
DEFAULT_FORMAT = "b[ext=mp4]/best[ext=mp4]/best"

METADATA_FIELDS = [
    "source",
    "nro",
    "category",
    "video",
    "url",
    "video_duration_sec",
    "behavior_start_sec",
    "behavior_end_sec",
    "event_duration_sec",
    "workbook_end_sec",
    "availability",
    "gender",
    "gender_label",
    "unique_subject",
    "timing_issue",
    "timing_note",
    "download_status",
    "download_error",
    "downloaded_path",
]


def col_to_idx(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Cannot parse cell reference: {cell_ref}")
    idx = 0
    for char in match.group(1):
        idx = idx * 26 + ord(char) - ord("A") + 1
    return idx - 1


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def parse_float(value: Any) -> float | None:
    text = clean_cell(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def youtube_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/")[0]
    query = parse_qs(parsed.query)
    if query.get("v"):
        return query["v"][0]
    return ""


def read_xlsx_rows(workbook_path: Path) -> list[dict[str, str]]:
    with ZipFile(workbook_path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{XLSX_NS}si"):
                text = "".join(t.text or "" for t in item.iter(f"{XLSX_NS}t"))
                shared_strings.append(text)

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        raw_rows: list[list[str]] = []
        for row in sheet.findall(f"{XLSX_NS}sheetData/{XLSX_NS}row"):
            sparse: dict[int, str] = {}
            for cell in row.findall(f"{XLSX_NS}c"):
                idx = col_to_idx(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_el = cell.find(f"{XLSX_NS}v")
                inline_el = cell.find(f"{XLSX_NS}is")
                value = ""
                if cell_type == "s" and value_el is not None:
                    value = shared_strings[int(value_el.text or "0")]
                elif cell_type == "inlineStr" and inline_el is not None:
                    value = "".join(t.text or "" for t in inline_el.iter(f"{XLSX_NS}t"))
                elif value_el is not None:
                    value = value_el.text or ""
                sparse[idx] = clean_cell(value)
            if sparse and any(v for v in sparse.values()):
                raw_rows.append([sparse.get(i, "") for i in range(max(sparse) + 1)])

    if not raw_rows:
        return []

    headers = [clean_cell(header) for header in raw_rows[0]]
    records = []
    for raw in raw_rows[1:]:
        padded = raw + [""] * max(0, len(headers) - len(raw))
        record = dict(zip(headers, padded[: len(headers)]))
        if record.get("Source") or record.get("Video") or record.get("URL"):
            records.append(record)
    return records


def normalize_record(record: dict[str, str]) -> dict[str, str]:
    start = parse_float(record.get("Start(s)"))
    workbook_end = parse_float(record.get("End(s)"))
    duration = parse_float(record.get("EventDuration(s)"))
    behavior_end = workbook_end
    timing_issue = False
    timing_note = ""

    if start is not None and duration is not None:
        derived_end = start + duration
        if workbook_end is None:
            behavior_end = derived_end
            timing_note = "end derived from start plus event duration"
        elif workbook_end < start or abs((workbook_end - start) - duration) > 0.25:
            behavior_end = derived_end
            timing_issue = True
            timing_note = "workbook end inconsistent; behavior_end_sec derived from start plus event duration"
    elif start is None and workbook_end is not None:
        timing_issue = True
        timing_note = "missing start with present workbook end"

    gender = clean_cell(record.get("Gender"))
    gender_label = {"M": "male", "F": "female"}.get(gender.upper(), "")
    availability = clean_cell(record.get("Availability"))

    return {
        "source": clean_cell(record.get("Source")),
        "nro": clean_cell(record.get("Nro")),
        "category": clean_cell(record.get("Category")),
        "video": clean_cell(record.get("Video")),
        "url": clean_cell(record.get("URL")),
        "video_duration_sec": fmt_float(parse_float(record.get("VideoDuration(s)"))),
        "behavior_start_sec": fmt_float(start),
        "behavior_end_sec": fmt_float(behavior_end),
        "event_duration_sec": fmt_float(duration),
        "workbook_end_sec": fmt_float(workbook_end),
        "availability": availability,
        "gender": gender,
        "gender_label": gender_label,
        "unique_subject": clean_cell(record.get("Unique Subject")),
        "timing_issue": "true" if timing_issue else "false",
        "timing_note": timing_note,
        "download_status": "",
        "download_error": "",
        "downloaded_path": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def install_import_path(extra_path: str | None) -> None:
    if extra_path:
        sys.path.insert(0, extra_path)


def download_video(
    row: dict[str, str],
    video_dir: Path,
    ydl: Any,
    url_cache: dict[str, Path],
    skip_existing: bool,
) -> None:
    video_name = slugify(row["video"])
    target_stem = video_dir / video_name
    existing = sorted(video_dir.glob(f"{video_name}.*"))
    if skip_existing and existing:
        row["download_status"] = "exists"
        row["downloaded_path"] = str(existing[0])
        url_cache.setdefault(row["url"], existing[0])
        return

    if row["url"] in url_cache and url_cache[row["url"]].exists():
        src = url_cache[row["url"]]
        dst = video_dir / f"{video_name}{src.suffix}"
        if src.resolve() != dst.resolve():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        row["download_status"] = "linked_duplicate_url"
        row["downloaded_path"] = str(dst)
        return

    before = set(video_dir.glob(f"{video_name}.*"))
    info = ydl.extract_info(row["url"], download=True)
    after = set(video_dir.glob(f"{video_name}.*"))
    created = sorted(after - before)
    if created:
        downloaded = created[0]
    else:
        downloaded = Path(ydl.prepare_filename(info))
    row["download_status"] = "downloaded"
    row["downloaded_path"] = str(downloaded)
    url_cache[row["url"]] = downloaded


def build_dataset(
    workbook: Path,
    out_dir: Path,
    metadata_only: bool,
    extra_import_path: str | None,
    ytdlp_format: str,
    skip_existing: bool,
    limit: int | None,
) -> dict[str, Any]:
    records = [normalize_record(record) for record in read_xlsx_rows(workbook)]
    timestamp = datetime.now(timezone.utc).isoformat()

    install_import_path(extra_import_path)
    ydl_cls = None
    if not metadata_only:
        try:
            import yt_dlp  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "yt-dlp is required for downloading. Install it or rerun with --metadata-only."
            ) from exc
        ydl_cls = yt_dlp.YoutubeDL

    out_dir.mkdir(parents=True, exist_ok=True)
    download_rows = [row for row in records if row["availability"].lower() != "unavailable"]
    if limit is not None:
        download_rows = download_rows[:limit]

    source_dirs: dict[str, Path] = {}
    for row in records:
        source = slugify(row["source"])
        source_dir = out_dir / source
        (source_dir / "videos").mkdir(parents=True, exist_ok=True)
        source_dirs[source] = source_dir

    if metadata_only:
        for row in records:
            if row["availability"].lower() == "unavailable":
                row["download_status"] = "skipped_workbook_unavailable"
            else:
                row["download_status"] = "not_downloaded_metadata_only"
    else:
        url_cache: dict[str, Path] = {}
        for row in records:
            if row["availability"].lower() == "unavailable":
                row["download_status"] = "skipped_workbook_unavailable"
                continue
            if row not in download_rows:
                row["download_status"] = "not_downloaded_limit"
                continue
            if not row["url"]:
                row["download_status"] = "failed"
                row["download_error"] = "missing URL"
                continue

            source_dir = source_dirs[slugify(row["source"])]
            video_dir = source_dir / "videos"
            ydl_opts = {
                "format": ytdlp_format,
                "outtmpl": str(video_dir / f"{slugify(row['video'])}.%(ext)s"),
                "ignoreerrors": False,
                "noplaylist": True,
                "quiet": False,
                "no_warnings": False,
                "restrictfilenames": True,
            }
            try:
                with ydl_cls(ydl_opts) as ydl:
                    download_video(row, video_dir, ydl, url_cache, skip_existing)
            except Exception as exc:  # yt-dlp emits many custom exception classes.
                row["download_status"] = "failed"
                row["download_error"] = str(exc).replace("\n", " ")[:1000]

    unavailable_rows = [
        row
        for row in records
        if row["download_status"] in {"skipped_workbook_unavailable", "failed"}
    ]
    timing_issue_rows = [row for row in records if row["timing_issue"] == "true"]

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_source[slugify(row["source"])].append(row)

    for source, rows in by_source.items():
        source_dir = out_dir / source
        write_csv(source_dir / "metadata.csv", rows, METADATA_FIELDS)
        write_json(source_dir / "metadata.json", rows)

    status_counts: dict[str, int] = defaultdict(int)
    for row in records:
        status_counts[row["download_status"]] += 1

    summary = {
        "created_at_utc": timestamp,
        "workbook": str(workbook),
        "out_dir": str(out_dir),
        "total_rows": len(records),
        "download_attempt_rows": len(download_rows) if not metadata_only else 0,
        "status_counts": dict(sorted(status_counts.items())),
        "sources": {source: len(rows) for source, rows in sorted(by_source.items())},
        "unavailable_or_failed_count": len(unavailable_rows),
        "timing_issue_count": len(timing_issue_rows),
        "format": ytdlp_format,
        "metadata_only": metadata_only,
    }

    write_csv(out_dir / "metadata.csv", records, METADATA_FIELDS)
    write_json(out_dir / "metadata.json", records)
    write_csv(out_dir / "download_report.csv", records, METADATA_FIELDS)
    write_csv(out_dir / "unavailable_videos.csv", unavailable_rows, METADATA_FIELDS)
    write_csv(out_dir / "timing_issues.csv", timing_issue_rows, METADATA_FIELDS)
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path(DEFAULT_WORKBOOK))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--extra-import-path", help="Extra sys.path entry, e.g. /tmp/asbd_deps")
    parser.add_argument("--format", default=DEFAULT_FORMAT, help="yt-dlp format selector")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit non-unavailable rows attempted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        workbook=args.workbook,
        out_dir=args.out_dir,
        metadata_only=args.metadata_only,
        extra_import_path=args.extra_import_path,
        ytdlp_format=args.format,
        skip_existing=not args.no_skip_existing,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
