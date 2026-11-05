# src/confusion_matrices_7class.py
"""Compute confusion matrices and generate LogNorm annotated heatmaps from 7-class prediction files.

Run from project root:
    python src/confusion_matrices_7class.py --models light_tcn_7class --subjects 1
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config
from config_7class import OUTPUT_DIR_7CLASS, EVENTS_6, EVENTS_7
from data_loading import load_series

MODEL_MODULES = {
    "logreg_csp_7class": "models.logreg_csp_7class", 
    "light_tcn_7class": "models.light_tcn_7class"
}


def load_saved_predictions(model_name, subject):
    npz_path = OUTPUT_DIR_7CLASS / f"preds_{model_name}_subj{subject}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {npz_path}")
    
    data = np.load(npz_path)
    preds = {int(k.replace("series", "")): data[k] for k in data.files}
    
    y_true_list = []
    y_score_list = []
    series_frames_info = {}
    
    for s in config.VAL_SERIES:
        _, y_true_s6 = load_series(subject, s)
        bg = (y_true_s6.sum(axis=0) == 0).astype(np.int8)[None, :]
        y_true_s7 = np.vstack([y_true_s6, bg]).T
        
        y_true_list.append(y_true_s7)
        y_score_list.append(preds[s])
        
        bg_mask = (y_true_s7[:, 6] == 1)
        series_frames_info[s] = {
            "total_frames": y_true_s7.shape[0],
            "background_frames": int(bg_mask.sum())
        }
        
    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)
    return y_true, y_score, series_frames_info


def save_heatmap(cm_df, title, filename, annot_size=9):
    plt.figure(figsize=(8, 6))
    max_val = max(cm_df.values.max(), 10)
    norm = mcolors.LogNorm(vmin=1, vmax=max_val)
    
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, 
                norm=norm, linewidths=.5, annot_kws={"size": annot_size})
                
    plt.title(title, fontsize=11, fontweight='bold', pad=12)
    plt.ylabel("True Label", fontsize=10, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def write_confusion_matrices_to_file(file_obj, model_name, identifier, y_true, y_score, threshold, cutoff, prefix_path, subject_id, series_audit_info):
    target_names = EVENTS_7

    print(f"\n  [Background Audit] Subject {subject_id} Breakdown:")
    total_subj_frames = 0
    total_subj_bg = 0
    for s, info in series_audit_info.items():
        t_frames = info["total_frames"]
        bg_frames = info["background_frames"]
        bg_pct = (bg_frames / t_frames) * 100 if t_frames > 0 else 0
        print(f"    - Series {s}: {bg_frames:,} background frames out of {t_frames:,} total frames ({bg_pct:.2f}% background)")
        total_subj_frames += t_frames
        total_subj_bg += bg_frames
    overall_bg_pct = (total_subj_bg / total_subj_frames) * 100 if total_subj_frames > 0 else 0
    print(f"    -> Subject {subject_id} Total: {total_subj_bg:,} background / {total_subj_frames:,} total ({overall_bg_pct:.2f}% background)")

    # (a) Binary 2x2 per event
    file_obj.write(f"=== Model: {model_name} | Target: {identifier} | (a) 2x2 Binary Matrices (7 Classes) ===\n\n")
    y_pred_bin = (y_score >= threshold).astype(int)
    for i, event_name in enumerate(target_names):
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        file_obj.write(f"Event: {event_name}\n  [[TN: {tn:6d}, FP: {fp:6d}],\n   [FN: {fn:6d}, TP: {tp:6d}]]\n")
        cm_2x2_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        save_heatmap(cm_2x2_df, f"2x2 Binary: {event_name}\n({model_name} - {identifier})", f"{prefix_path}_matrix_a_{event_name}_heatmap.png", annot_size=11)

    # (b) 7-Class Non-Overlap (with dropped frame logging)
    file_obj.write(f"\n=== (b) 7-Class Confusion Matrix (Non-Overlap Frames Only) ===\n\n")
    n_frames = y_true.shape[0]
    
    # Check overlap across the 6 gesture columns (ignoring background index 6)
    active_counts = y_true[:, :6].sum(axis=1)
    non_overlap_mask = active_counts <= 1
    dropped_count = n_frames - non_overlap_mask.sum()
    dropped_pct = (dropped_count / n_frames) * 100 if n_frames > 0 else 0
    
    print(f"  [Non-Overlap Audit] Total frames: {n_frames:,}, Kept: {non_overlap_mask.sum():,}, Dropped overlap: {dropped_count:,} ({dropped_pct:.2f}%)")
    file_obj.write(f"Total frames: {n_frames}, Kept: {non_overlap_mask.sum()}, Dropped overlap: {dropped_count} ({dropped_pct:.2f}%)\n\n")
    
    y_tf, y_sf = y_true[non_overlap_mask], y_score[non_overlap_mask]
    true_c = y_tf.argmax(axis=1)
    pred_c = y_sf.argmax(axis=1)
    
    cm_b = confusion_matrix(true_c, pred_c, labels=list(range(len(target_names))))
    cm_b_df = pd.DataFrame(cm_b, index=target_names, columns=target_names)
    file_obj.write(cm_b_df.to_string() + "\n")
    save_heatmap(cm_b_df, f"7-Class Non-Overlap Matrix (Log Scale)\n({model_name} - {identifier})", f"{prefix_path}_matrix_b_heatmap.png")

    # (c) 7-Class Nearest-Instant
    file_obj.write(f"\n=== (c) 7-Class Confusion Matrix (Nearest-Instant, cutoff={cutoff}) ===\n\n")
    onsets_per_event = [np.where(np.diff(y_true[:, i], prepend=0) == 1)[0] for i in range(6)]
    true_c_c = np.full(n_frames, 6, dtype=int)
    for t in range(n_frames):
        min_dist, closest = float('inf'), 6
        for i in range(6):
            if len(onsets_per_event[i]) > 0:
                d_min = np.abs(onsets_per_event[i] - t).min()
                if d_min < min_dist:
                    min_dist, closest = d_min, i
        true_c_c[t] = closest if min_dist <= cutoff else 6
        
    pred_c_c = y_score.argmax(axis=1)
    
    cm_c = confusion_matrix(true_c_c, pred_c_c, labels=list(range(len(target_names))))
    cm_c_df = pd.DataFrame(cm_c, index=target_names, columns=target_names)
    file_obj.write(cm_c_df.to_string() + "\n")
    save_heatmap(cm_c_df, f"7-Class Nearest-Instant Matrix (Log Scale)\n({model_name} - {identifier})", f"{prefix_path}_matrix_c_heatmap.png")
    
    print(f"  -> Saved all LogNorm heatmaps for {identifier} in {OUTPUT_DIR_7CLASS.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="Compute 7-class confusion matrices from 7-class prediction files.")
    ap.add_argument("--models", nargs="+", default=list(MODEL_MODULES), choices=list(MODEL_MODULES))
    ap.add_argument("--subjects", nargs="+", type=int, default=[1])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cutoff", type=int, default=75)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    OUTPUT_DIR_7CLASS.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if args.aggregate:
            print(f"\n--- Processing AGGREGATE 7-class heatmaps for model: {model_name} ---")
            all_true, all_score = [], []
            combined_series_audit = {}
            
            for subj in args.subjects:
                y_t, y_s, subj_audit = load_saved_predictions(model_name, subj)
                all_true.append(y_t)
                all_score.append(y_s)
                for s, info in subj_audit.items():
                    combined_series_audit[f"Subj{subj}_Ser{s}"] = info
            
            y_true_agg = np.concatenate(all_true, axis=0)
            y_score_agg = np.concatenate(all_score, axis=0)
            
            output_file = OUTPUT_DIR_7CLASS / f"confusion_matrices_{model_name}_aggregated.txt"
            prefix_path = str(OUTPUT_DIR_7CLASS / f"confusion_matrices_{model_name}_aggregated")
            
            with open(output_file, "w", encoding="utf-8") as f:
                write_confusion_matrices_to_file(f, model_name, f"Aggregated {args.subjects}", y_true_agg, y_score_agg, args.threshold, args.cutoff, prefix_path, "All Subjects", combined_series_audit)
            
        else:
            for subj in args.subjects:
                print(f"\n--- Processing 7-class heatmaps for {model_name}, Subject {subj} ---")
                y_true, y_score, subj_audit = load_saved_predictions(model_name, subj)
                
                output_file = OUTPUT_DIR_7CLASS / f"confusion_matrices_{model_name}_subj{subj}.txt"
                prefix_path = str(OUTPUT_DIR_7CLASS / f"confusion_matrices_{model_name}_subj{subj}")
                
                with open(output_file, "w", encoding="utf-8") as f:
                    write_confusion_matrices_to_file(f, model_name, f"Subject {subj}", y_true, y_score, args.threshold, args.cutoff, prefix_path, subj, subj_audit)


# if __name__ == "__main__":
main()