#claude generated
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

import config


def dense_metrics(y_true, y_score, thresh=0.5):
    """y_true, y_score: (n_steps, 6). Returns per-event and macro metrics.

    UAR depends on the threshold. With rare events most probabilities stay
    below 0.5, so check how the Rasanen paper sets it before comparing.
    """
    out, aucs, aps, uars = {}, [], [], []
    y_pred = (y_score >= thresh).astype(int)
    for i, name in enumerate(config.EVENTS):
        y = y_true[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            auc = ap = uar = np.nan
        else:
            auc = roc_auc_score(y, y_score[:, i])
            ap = average_precision_score(y, y_score[:, i])
            uar = balanced_accuracy_score(y, y_pred[:, i])
        out[f"auc/{name}"], out[f"ap/{name}"] = auc, ap
        aucs.append(auc); aps.append(ap); uars.append(uar)
    out["auc_macro"] = float(np.nanmean(aucs))    # the Kaggle metric
    out["mAP"] = float(np.nanmean(aps))
    out["UAR"] = float(np.nanmean(uars))
    return out


def wandb_start(model_name, cfg):
    """One W&B run per model. Subjects are logged as steps of that run."""
    import wandb
    return wandb.init(project="biohdc-dense-labeling", group="baselines",
                      name=model_name, config=cfg)


def wandb_log_subject(run, subject, metrics, info):
    run.log({"subject": subject, **metrics, **info}, step=subject)


def wandb_finish(run, df):
    """df: this model's rows, one per subject. Adds mean summary numbers and a table."""
    import wandb
    summary = {f"mean/{c}": float(df[c].mean()) for c in ["auc_macro", "mAP", "UAR"]}
    summary.update({c: float(df[c].iloc[0]) for c in ["params", "stored_bytes", "ops_per_second"]})
    run.summary.update(summary)
    table_cols = ["subject", "auc_macro", "mAP", "UAR"] + [c for c in df.columns if c.startswith("auc/")]
    run.log({"per_subject": wandb.Table(dataframe=df[table_cols])})
    run.finish()
# import numpy as np
# from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
# import wandb
# import config


# def dense_metrics(y_true, y_score, thresh=0.5):
#     """y_true, y_score: (n_steps, 6). Returns per-event and macro metrics.

#     UAR depends on the threshold. With rare events most probabilities stay
#     below 0.5, so check how the Rasanen paper sets it before comparing.
#     """
#     out, aucs, aps, uars = {}, [], [], []
#     y_pred = (y_score >= thresh).astype(int)
#     for i, name in enumerate(config.EVENTS):
#         y = y_true[:, i]
#         if y.sum() == 0 or y.sum() == len(y):
#             auc = ap = uar = np.nan
#         else:
#             auc = roc_auc_score(y, y_score[:, i])
#             ap = average_precision_score(y, y_score[:, i])
#             uar = balanced_accuracy_score(y, y_pred[:, i])
#         out[f"auc/{name}"], out[f"ap/{name}"] = auc, ap
#         aucs.append(auc); aps.append(ap); uars.append(uar)
#     out["auc_macro"] = float(np.nanmean(aucs))    # the Kaggle metric
#     out["mAP"] = float(np.nanmean(aps))
#     out["UAR"] = float(np.nanmean(uars))
#     return out


# def log_wandb(model_name, subject, metrics, info, cfg, noise_sigma=0.0):
    
#     run = wandb.init(project="biohdc-dense-labeling", group=model_name,
#                      name=f"{model_name}_subj{subject}_noise{noise_sigma}",
#                      config=cfg, reinit=True)
#     wandb.log({**metrics, **info, "noise_sigma": noise_sigma, "subject": subject})
#     run.finish()