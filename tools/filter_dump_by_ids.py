#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dump", type=str)
    ap.add_argument("ids_txt", type=str)
    ap.add_argument("out_dump", type=str)
    args = ap.parse_args()

    dump = pickle.load(open(args.in_dump, "rb"))
    samples = dump["samples"]

    keep_ids = set([ln.strip() for ln in Path(args.ids_txt).read_text().splitlines() if ln.strip()])
    out_samples = [s for s in samples if s.get("sample_id") in keep_ids]

    out = dict(dump)
    out["samples"] = out_samples
    out["eval_ids_source"] = str(Path(args.ids_txt).resolve())
    out["eval_num_ids"] = len(keep_ids)
    out["eval_num_samples_kept"] = len(out_samples)

    Path(args.out_dump).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(args.out_dump, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] in_samples={len(samples)} keep_ids={len(keep_ids)} out_samples={len(out_samples)}")
    print(f"[INFO] wrote {args.out_dump}")

if __name__ == "__main__":
    main()
