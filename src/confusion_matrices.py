# generated for thesis baseline analysis
"""Compute confusion matrices and generate annotated heatmaps (including 2x2 per event) from saved .npz files.

Run from project root:
    python src/confusion_matrices.py --models light_tcn --subjects 1
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config
from data_loading import load_series

MODEL_MODULES = {"logreg_csp": "models.logreg_csp", "light_tcn": "models.light_tcn"}


def load_saved_predictions(model_name, subject):
    """Loads and validates predictions from the pre-saved .npz file in OUTPUT_DIR."""
    npz_path = config.OUTPUT_DIR / f"preds_{model_name}_subj{subject}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {npz_path}")
    
    data = np.load(npz_path)
    preds = {int(k.replace("series", "")): data[k] for k in data.files}
    
    y_true_list = []
    y_score_list = []
    series_frames_info = {} # Track frame counts per series for auditing
    
    for s in config.VAL_SERIES:
        _, y_true_s = load_series(subject, s)
        # y_true_s is shape (6, N), transpose to (N, 6)
        y_true_t = y_true_s.T
        y_true_list.append(y_true_t)
        y_score_list.append(preds[s])
        
        # Calculate background frames for this specific series (where all 6 binary targets are 0)
        bg_mask = (y_true_t.sum(axis=1) == 0)
        series_frames_info[s] = {
            "total_frames": y_true_t.shape[0],
            "background_frames": int(bg_mask.sum())
        }
        
    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)
    return y_true, y_score, series_frames_info


# def save_heatmap(cm_df, title, filename, annot_size=9):
#     """Helper function to plot, annotate, and save a confusion matrix heatmap."""
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, 
#                 linewidths=.5, annot_kws={"size": annot_size})
#     plt.title(title, fontsize=11, fontweight='bold', pad=12)
#     plt.ylabel("True Label", fontsize=10, fontweight='bold')
#     plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()

import matplotlib.colors as mcolors # Make sure this is imported at the top of your file

def save_heatmap(cm_df, title, filename, annot_size=9):
    """Helper function to plot, annotate, and save a confusion matrix heatmap using a log color scale."""
    plt.figure(figsize=(8, 6))
    
    # Use LogNorm so counts < 10k get visible color contrast against massive counts (> 100k)
    # Note: LogNorm requires values > 0, so we use vmin=1
    norm = mcolors.LogNorm(vmin=1, vmax=max(cm_df.values.max(), 10))
    
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
    target_names = list(config.EVENTS) + ["Background"]

    # --- Print audit summary for background frames per series (for your sanity check) ---
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

    # --- (a) 2x2 Binary Matrices per Event ---
    file_obj.write(f"=== Model: {model_name} | Target: {identifier} | (a) 2x2 Binary Matrices ===\n")
    file_obj.write("INSIGHT: Evaluates each event independently. Unaffected by class overlap.\n\n")
    y_pred_bin = (y_score >= threshold).astype(int)
    
    for i, event_name in enumerate(config.EVENTS):
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        file_obj.write(f"Event: {event_name}\n  [[TN: {tn:6d}, FP: {fp:6d}],\n   [FN: {fn:6d}, TP: {tp:6d}]]\n")

        # Save individual 2x2 heatmap for each event
        cm_2x2_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        heatmap_path_2x2 = f"{prefix_path}_matrix_a_{event_name}_heatmap.png"
        save_heatmap(cm_2x2_df, f"2x2 Binary Matrix: {event_name}\n({model_name} - {identifier})", heatmap_path_2x2, annot_size=11)

    # --- (b) 7-Class Non-Overlap ---
    file_obj.write(f"\n=== (b) 7-Class Confusion Matrix (Non-Overlap Frames Only) ===\n")
    file_obj.write("INSIGHT: Multi-class evaluation dropping overlap frames. Biased towards easy static periods.\n\n")
    n_frames, n_events = y_true.shape
    active_counts = y_true.sum(axis=1)
    non_overlap_mask = active_counts <= 1
    dropped_count = n_frames - non_overlap_mask.sum()
    file_obj.write(f"Total frames: {n_frames}, Kept: {non_overlap_mask.sum()}, Dropped overlap: {dropped_count} ({dropped_count/n_frames:.2%})\n")
    
    y_tf, y_sf = y_true[non_overlap_mask], y_score[non_overlap_mask]
    true_c = np.full(y_tf.shape[0], len(config.EVENTS), dtype=int)
    for i in range(n_events):
        true_c[y_tf[:, i] == 1] = i
    max_s = y_sf.max(axis=1)
    pred_c = np.full(y_sf.shape[0], len(config.EVENTS), dtype=int)
    valid = max_s >= threshold
    pred_c[valid] = y_sf[valid].argmax(axis=1)
    
    cm_b = confusion_matrix(true_c, pred_c, labels=list(range(len(target_names))))
    cm_b_df = pd.DataFrame(cm_b, index=target_names, columns=target_names)
    file_obj.write(cm_b_df.to_string() + "\n")

    # Save heatmap for (b)
    heatmap_path_b = f"{prefix_path}_matrix_b_heatmap.png"
    save_heatmap(cm_b_df, f"7-Class Non-Overlap Matrix\n({model_name} - {identifier})", heatmap_path_b)

    # --- (c) Nearest-Instant Labelling ---
    file_obj.write(f"\n=== (c) 7-Class Confusion Matrix (Nearest-Instant, cutoff={cutoff}) ===\n")
    file_obj.write("INSIGHT: Mutually exclusive 7-class assignment based on closest event onset without dropping frames.\n\n")
    onsets_per_event = [np.where(np.diff(y_true[:, i], prepend=0) == 1)[0] for i in range(n_events)]
    true_c_c = np.full(n_frames, len(config.EVENTS), dtype=int)
    for t in range(n_frames):
        min_dist, closest = float('inf'), len(config.EVENTS)
        for i in range(n_events):
            if len(onsets_per_event[i]) > 0:
                d_min = np.abs(onsets_per_event[i] - t).min()
                if d_min < min_dist:
                    min_dist, closest = d_min, i
        true_c_c[t] = closest if min_dist <= cutoff else len(config.EVENTS)
        
    pred_c_c = np.full(n_frames, len(config.EVENTS), dtype=int)
    valid_c = y_score.max(axis=1) >= threshold
    pred_c_c[valid_c] = y_score[valid_c].argmax(axis=1)
    
    cm_c = confusion_matrix(true_c_c, pred_c_c, labels=list(range(len(target_names))))
    cm_c_df = pd.DataFrame(cm_c, index=target_names, columns=target_names)
    file_obj.write(cm_c_df.to_string() + "\n")

    # Save heatmap for (c)
    heatmap_path_c = f"{prefix_path}_matrix_c_heatmap.png"
    save_heatmap(cm_c_df, f"7-Class Nearest-Instant Matrix\n({model_name} - {identifier})", heatmap_path_c)
    
    print(f"  -> Saved all heatmaps (2x2 event grids, non-overlap, and nearest-instant) for {identifier}.")


def main():
    ap = argparse.ArgumentParser(description="Compute confusion matrices and heatmaps from saved .npz files.")
    ap.add_argument("--models", nargs="+", default=list(MODEL_MODULES), choices=list(MODEL_MODULES))
    ap.add_argument("--subjects", nargs="+", type=int, default=[1])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cutoff", type=int, default=75)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    config.OUTPUT_DIR.mkdir(exist_ok=True)

    for model_name in args.models:
        if args.aggregate:
            print(f"\n--- Processing AGGREGATE heatmaps for model: {model_name} ---")
            all_true, all_score = [], []
            combined_series_audit = {}
            
            for subj in args.subjects:
                y_t, y_s, subj_audit = load_saved_predictions(model_name, subj)
                all_true.append(y_t)
                all_score.append(y_s)
                for s, info in subj_audit.items():
                    key = f"Subj{subj}_Ser{s}"
                    combined_series_audit[key] = info
            
            y_true_agg = np.concatenate(all_true, axis=0)
            y_score_agg = np.concatenate(all_score, axis=0)
            
            output_file = config.OUTPUT_DIR / f"confusion_matrices_{model_name}_aggregated.txt"
            prefix_path = str(config.OUTPUT_DIR / f"confusion_matrices_{model_name}_aggregated")
            
            with open(output_file, "w", encoding="utf-8") as f:
                write_confusion_matrices_to_file(f, model_name, f"Aggregated {args.subjects}", y_true_agg, y_score_agg, args.threshold, args.cutoff, prefix_path, "All Subjects", combined_series_audit)
            print(f"  -> Saved aggregate text log and heatmaps to {config.OUTPUT_DIR.resolve()}")
            
        else:
            for subj in args.subjects:
                print(f"\n--- Processing heatmaps for {model_name}, Subject {subj} ---")
                y_true, y_score, subj_audit = load_saved_predictions(model_name, subj)
                
                output_file = config.OUTPUT_DIR / f"confusion_matrices_{model_name}_subj{subj}.txt"
                prefix_path = str(config.OUTPUT_DIR / f"confusion_matrices_{model_name}_subj{subj}")
                
                with open(output_file, "w", encoding="utf-8") as f:
                    write_confusion_matrices_to_file(f, model_name, f"Subject {subj}", y_true, y_score, args.threshold, args.cutoff, prefix_path, subj, subj_audit)
                print(f"  -> Saved text log to {output_file.resolve()}")


# if __name__ == "__main__":
main()