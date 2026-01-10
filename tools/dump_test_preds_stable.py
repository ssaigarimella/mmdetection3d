#!/usr/bin/env python3
import argparse
import os
import pickle
import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default="./work_dirs/_dump_preds_runner")
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--nms-thr", type=float, default=None)
    ap.add_argument("--max-samples", type=int, default=-1)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    # Force low thresholds at the MODEL level (this is what matters)
    if isinstance(cfg.model, dict):
        cfg.model.setdefault("test_cfg", {})
        cfg.model["test_cfg"]["score_thr"] = float(args.score_thr)
        if args.nms_thr is not None:
            cfg.model["test_cfg"]["nms_thr"] = float(args.nms_thr)

    # IMPORTANT: do NOT set cfg.test_evaluator=None (Runner will error).
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = runner.model
    model.eval()

    test_loader = runner.build_dataloader(cfg.test_dataloader)

    preds = []
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            out = model.test_step(data_batch)  # list[Det3DDataSample]
            preds.extend(out)
            if args.max_samples > 0 and len(preds) >= args.max_samples:
                preds = preds[:args.max_samples]
                break

    with open(args.out, "wb") as f:
        pickle.dump(preds, f)

    print("Wrote:", os.path.abspath(args.out))
    print("Frames:", len(preds))

if __name__ == "__main__":
    main()
