#!/usr/bin/env python3
"""
tools/print_metrics_table.py

Reads tools/metrics/*.txt produced by tools/eval_lidar_ap_from_dump.py and prints
clean comparison tables to the terminal.

Adds "paper-style" grouped tables (similar organization to DAIR-V2X table style):
- Methods (run_tag) as rows
- Classes grouped as columns (with canonical IoU per class shown in header)
- Computes mAP across available classes (excluding "no GT")

Keeps original output tables unchanged.

Edit METRICS_DIR below if needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# -------------------------
# IN-CODE PARAMS
# -------------------------
METRICS_DIR = Path("/home/dellg16ssg/mmdetection3d/tools/metrics")
GLOB_PATTERN = "*.txt"

RUN_ORDER = ["vehicle", "infra", "late_fusion"]

# Classes to exclude from display
EXCLUDE_CLASSES = {"Cyclist"}


# -------------------------
# Parsing regex
# -------------------------
RE_RUN_TAG = re.compile(r"^\[INFO\]\s+run_tag:\s*(\S+)\s*$")
RE_SECTION = re.compile(r"^===\s*(STRICT|LOOSE)\s*===\s*$")
RE_NO_GT   = re.compile(r"^(\w+):\s*no\s+GT\s*$", re.IGNORECASE)
RE_METRIC  = re.compile(
    r"^(?P<cls>\w+)\s+IoU=(?P<iou>[0-9.]+)\s+\|\s+"
    r"BEV\s+AP11=(?P<bev11>[0-9.]+)\s+AP40=(?P<bev40>[0-9.]+)\s+\|\s+"
    r"3D\s+AP11=(?P<d311>[0-9.]+)\s+AP40=(?P<d340>[0-9.]+)\s*$"
)


def infer_run_tag_from_name(p: Path) -> str:
    s = p.name.lower()
    if "late" in s and "fusion" in s:
        return "late_fusion"
    if "infra" in s:
        return "infra"
    if "vehicle" in s:
        return "vehicle"
    return "unknown"


def parse_metrics_file(path: Path) -> Dict[str, Any]:
    txt = path.read_text().splitlines()

    current_section: Optional[str] = None  # STRICT / LOOSE
    out: Dict[str, Any] = {
        "file": str(path),
        "run_tag": None,
        "sections": {"STRICT": {}, "LOOSE": {}},  # cls -> metrics dict
    }

    for line in txt:
        line = line.strip()

        m = RE_RUN_TAG.match(line)
        if m:
            out["run_tag"] = m.group(1).strip()
            continue

        m = RE_SECTION.match(line)
        if m:
            current_section = m.group(1)
            continue

        if current_section is None:
            continue

        m = RE_NO_GT.match(line)
        if m:
            cls = m.group(1)
            if cls in EXCLUDE_CLASSES:
                continue
            out["sections"][current_section][cls] = {"no_gt": True, "iou": None}
            continue

        m = RE_METRIC.match(line)
        if m:
            cls = m.group("cls")
            if cls in EXCLUDE_CLASSES:
                continue
            out["sections"][current_section][cls] = {
                "no_gt": False,
                "iou": float(m.group("iou")),
                "bev_ap11": float(m.group("bev11")),
                "bev_ap40": float(m.group("bev40")),
                "d3_ap11": float(m.group("d311")),
                "d3_ap40": float(m.group("d340")),
            }
            continue

    if out["run_tag"] is None:
        out["run_tag"] = infer_run_tag_from_name(path)

    return out


# -------------------------
# Pretty printing
# -------------------------
def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def print_table(title: str, rows: List[List[str]], headers: List[str]) -> None:
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def line(sep: str = "-") -> str:
        return "+ " + " + ".join(sep * w for w in widths) + " +"

    def row(cells: List[str]) -> str:
        return "| " + " | ".join(str(cells[i]).ljust(widths[i]) for i in range(len(widths))) + " |"

    print("\n" + title)
    print(line("-"))
    print(row(headers))
    print(line("="))
    for r in rows:
        print(row(r))
    print(line("-"))


def path_basename(p: str) -> str:
    try:
        return Path(p).name
    except Exception:
        return p


# -------------------------
# New: paper-style grouped summary (NO source_file column)
# -------------------------
def _canonical_iou_per_class(parsed: List[Dict[str, Any]], section: str, classes: List[str]) -> Dict[str, Optional[float]]:
    """
    Pick a canonical IoU per class for header display, per section.
    If multiple runs disagree, keep the first seen non-None value.
    """
    out: Dict[str, Optional[float]] = {c: None for c in classes}
    for d in parsed:
        sec = d["sections"].get(section, {})
        for c in classes:
            m = sec.get(c)
            if not m or m.get("no_gt", False):
                continue
            if out[c] is None and m.get("iou") is not None:
                out[c] = float(m["iou"])
    return out


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _compute_map(row_metrics: Dict[str, Dict[str, Any]], classes: List[str], key: str) -> Optional[float]:
    vals: List[float] = []
    for c in classes:
        m = row_metrics.get(c)
        if not m or m.get("no_gt", False):
            continue
        v = m.get(key, None)
        if isinstance(v, float):
            vals.append(v)
    return _mean(vals)


def print_paper_style_tables(parsed: List[Dict[str, Any]], classes: List[str]) -> None:
    """
    Prints extra tables closer to "paper table organization":
    - One wide BEV table and one wide 3D table per section.
    - Columns grouped by class, with canonical IoU per class included in header.
    - Adds mAP across classes (AP11 and AP40) per run_tag.
    - Does NOT print source file names in these paper-style tables.
    """
    for sec in ("STRICT", "LOOSE"):
        canon_iou = _canonical_iou_per_class(parsed, sec, classes)

        # Wide BEV table
        bev_headers: List[str] = ["run_tag"]
        for c in classes:
            iou = canon_iou.get(c)
            iou_s = _fmt(iou) if iou is not None else "-"
            bev_headers += [f"{c} (IoU={iou_s}) AP11", f"{c} (IoU={iou_s}) AP40"]
        bev_headers += ["mAP AP11", "mAP AP40"]

        bev_rows: List[List[str]] = []
        for d in parsed:
            rt = d["run_tag"]
            sec_metrics: Dict[str, Dict[str, Any]] = d["sections"].get(sec, {})
            row: List[str] = [rt]

            for c in classes:
                m = sec_metrics.get(c)
                if not m:
                    row += ["-", "-"]
                elif m.get("no_gt", False):
                    row += ["no GT", "no GT"]
                else:
                    row += [_fmt(m.get("bev_ap11")), _fmt(m.get("bev_ap40"))]

            map11 = _compute_map(sec_metrics, classes, "bev_ap11")
            map40 = _compute_map(sec_metrics, classes, "bev_ap40")
            row += [_fmt(map11), _fmt(map40)]
            bev_rows.append(row)

        print_table(
            title=f"{sec} | BEV AP (paper-style: classes grouped, with mAP over classes)",
            headers=bev_headers,
            rows=bev_rows,
        )

        # Wide 3D table
        d3_headers: List[str] = ["run_tag"]
        for c in classes:
            iou = canon_iou.get(c)
            iou_s = _fmt(iou) if iou is not None else "-"
            d3_headers += [f"{c} (IoU={iou_s}) AP11", f"{c} (IoU={iou_s}) AP40"]
        d3_headers += ["mAP AP11", "mAP AP40"]

        d3_rows: List[List[str]] = []
        for d in parsed:
            rt = d["run_tag"]
            sec_metrics = d["sections"].get(sec, {})
            row = [rt]

            for c in classes:
                m = sec_metrics.get(c)
                if not m:
                    row += ["-", "-"]
                elif m.get("no_gt", False):
                    row += ["no GT", "no GT"]
                else:
                    row += [_fmt(m.get("d3_ap11")), _fmt(m.get("d3_ap40"))]

            map11 = _compute_map(sec_metrics, classes, "d3_ap11")
            map40 = _compute_map(sec_metrics, classes, "d3_ap40")
            row += [_fmt(map11), _fmt(map40)]
            d3_rows.append(row)

        print_table(
            title=f"{sec} | 3D AP (paper-style: classes grouped, with mAP over classes)",
            headers=d3_headers,
            rows=d3_rows,
        )

        # Compact "Overall-like" table: just the mAP numbers per run_tag
        compact_headers = ["run_tag", "BEV mAP11", "BEV mAP40", "3D mAP11", "3D mAP40"]
        compact_rows: List[List[str]] = []
        for d in parsed:
            rt = d["run_tag"]
            sec_metrics = d["sections"].get(sec, {})
            bev_map11 = _compute_map(sec_metrics, classes, "bev_ap11")
            bev_map40 = _compute_map(sec_metrics, classes, "bev_ap40")
            d3_map11  = _compute_map(sec_metrics, classes, "d3_ap11")
            d3_map40  = _compute_map(sec_metrics, classes, "d3_ap40")
            compact_rows.append([
                rt,
                _fmt(bev_map11), _fmt(bev_map40),
                _fmt(d3_map11),  _fmt(d3_map40),
            ])

        print_table(
            title=f"{sec} | Overall-like summary (mAP over classes)",
            headers=compact_headers,
            rows=compact_rows,
        )


def main():
    if not METRICS_DIR.is_dir():
        raise SystemExit(f"ERROR: METRICS_DIR does not exist: {METRICS_DIR}")

    files = sorted(METRICS_DIR.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"ERROR: No files matched {METRICS_DIR / GLOB_PATTERN}")

    parsed = [parse_metrics_file(p) for p in files]

    def run_sort_key(rt: str) -> Tuple[int, str]:
        return (RUN_ORDER.index(rt) if rt in RUN_ORDER else 999, rt)

    parsed.sort(key=lambda d: run_sort_key(d["run_tag"]))

    # Collect classes seen (excluding Cyclist)
    classes: List[str] = []
    for d in parsed:
        for sec in ("STRICT", "LOOSE"):
            for cls in d["sections"][sec].keys():
                if cls in EXCLUDE_CLASSES:
                    continue
                if cls not in classes:
                    classes.append(cls)

    # Stable preference order if present
    pref = ["Car", "Pedestrian"]
    classes = [c for c in pref if c in classes] + [c for c in classes if c not in pref]

    # -------------------------
    # Original outputs (UNCHANGED, still include source_file)
    # -------------------------
    for sec in ("STRICT", "LOOSE"):
        # BEV table
        bev_rows: List[List[str]] = []
        for d in parsed:
            rt = d["run_tag"]
            for cls in classes:
                m = d["sections"][sec].get(cls)
                if not m:
                    continue
                if m.get("no_gt", False):
                    bev_rows.append([rt, cls, "-", "no GT", "no GT", path_basename(d["file"])])
                else:
                    bev_rows.append([rt, cls, _fmt(m["iou"]), _fmt(m["bev_ap11"]), _fmt(m["bev_ap40"]), path_basename(d["file"])])

        print_table(
            title=f"{sec} | BEV AP (IoU threshold shown per class)",
            headers=["run_tag", "class", "IoU", "AP11", "AP40", "source_file"],
            rows=bev_rows,
        )

        # 3D table
        d3_rows: List[List[str]] = []
        for d in parsed:
            rt = d["run_tag"]
            for cls in classes:
                m = d["sections"][sec].get(cls)
                if not m:
                    continue
                if m.get("no_gt", False):
                    d3_rows.append([rt, cls, "-", "no GT", "no GT", path_basename(d["file"])])
                else:
                    d3_rows.append([rt, cls, _fmt(m["iou"]), _fmt(m["d3_ap11"]), _fmt(m["d3_ap40"]), path_basename(d["file"])])

        print_table(
            title=f"{sec} | 3D AP (IoU threshold shown per class)",
            headers=["run_tag", "class", "IoU", "AP11", "AP40", "source_file"],
            rows=d3_rows,
        )

    # -------------------------
    # New outputs (paper-style; NO source_file)
    # -------------------------
    print_paper_style_tables(parsed, classes)


if __name__ == "__main__":
    main()
