# src/main_7class.py
"""Run 7-class baseline models and save outputs cleanly into outputs/7_class_outputs/

Run from project root:
    python src/main_7class.py --models logreg_csp_7class light_tcn_7class --subjects 1
"""
import argparse
import importlib
import numpy as np
import pandas as pd

import config
from config_7class import OUTPUT_DIR_7CLASS
from data_loading import load_series
from metrics import dense_metrics, wandb_start, wandb_log_subject, wandb_finish

MODEL_MODULES = {
    "logreg_csp_7class": "models.logreg_csp_7class", 
    "light_tcn_7class": "models.light_tcn_7class"
}


def evaluate(model_name, subject):
    module = importlib.import_module(MODEL_MODULES[model_name])
    preds, info = module.run(subject, config.TRAIN_SERIES, config.VAL_SERIES)

    # Load 7-class ground truth
    y_true_list = []
    for s in config.VAL_SERIES:
        _, y6 = load_series(subject, s)
        bg = (y6.sum(axis=0) == 0).astype(np.int8)[None, :]
        y7 = np.vstack([y6, bg]).T
        y_true_list.append(y7)

    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate([preds[s] for s in config.VAL_SERIES], axis=0)
    
    assert y_true.shape == y_score.shape, (y_true.shape, y_score.shape)
    
    # Calculate metrics across 6 gestures (for direct baseline comparison)
    metrics = dense_metrics(y_true[:, :6], y_score[:, :6])
    return metrics, info, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_MODULES), choices=list(MODEL_MODULES))
    ap.add_argument("--subjects", nargs="+", type=int, default=[1])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--check-causality", action="store_true",
                        help="only test that predictions never use future samples, then exit")
    args = ap.parse_args()

    # Competition rule: a prediction may not use frames after the one being predicted.
    all_ok = True
    for name in args.models:
        module = importlib.import_module(MODEL_MODULES[name])
        check = getattr(module, "check_causality", None)
        if check is None:
            print(f"[causality] {name}: no check defined")
        else:
            all_ok = check() and all_ok
    if not all_ok:
        raise SystemExit("Causality check failed: a model uses future data. Fix it before training.")
    if args.check_causality:
        return
    OUTPUT_DIR_7CLASS.mkdir(parents=True, exist_ok=True)
    rows = []
    
    for name in args.models:
        run = None
        if args.wandb:
            run = wandb_start(name, {"model": name, "protocol": "within-subject-7class",
                                     "train": config.TRAIN_SERIES, "val": config.VAL_SERIES})
        model_rows = []
        for subject in sorted(args.subjects):
            print(f"== {name}, subject {subject}: train {config.TRAIN_SERIES}, val {config.VAL_SERIES}")
            metrics, info, preds = evaluate(name, subject)
            
            # Save 7-class predictions into outputs/7_class_outputs/
            np.savez_compressed(OUTPUT_DIR_7CLASS / f"preds_{name}_subj{subject}.npz",
                                **{f"series{s}": p for s, p in preds.items()})
            if run is not None:
                wandb_log_subject(run, subject, metrics, info)
            model_rows.append({"model": name, "subject": subject, **metrics, **info})
        if run is not None:
            wandb_finish(run, pd.DataFrame(model_rows))
        rows += model_rows

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR_7CLASS / "results_7class.csv", index=False)
    cols = ["model", "subject", "auc_macro", "mAP", "UAR", "params", "stored_bytes", "ops_per_second"]
    print("\n--- 7-Class Baseline Results ---")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()