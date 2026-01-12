#!/usr/bin/env python3
"""
tools/print_metrics_table.py

Reads tools/metrics/*.txt produced by tools/eval_lidar_ap_from_dump.py and prints
clean comparison tables to the terminal.

Modifications vs prior version:
- Drops Cyclist entirely.
- Adds IoU threshold column (per class line: "IoU=...") to the tables.

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

if __name__ == "__main__":
    main()
