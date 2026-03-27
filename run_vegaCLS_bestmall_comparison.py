"""
run_cls_experiment.py
=====================
Loops over positive-class dataset sizes and pretrained encoder checkpoints,
trains LinearSVC / CatBoost classifiers on CLS tokens (and BestMall GP
features), logs all performance metrics, and saves CLS arrays for later
plotting.

Usage
-----
    python run_cls_experiment.py [--output_dir RESULTS] [--plot]

All paths that are collected in the CONFIG block at the top – edit there
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  –  edit paths here
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = dict(
    # ── dataset ───────────────────────────────────────────────────────────────
    data_root        = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like",
    test_parq        = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like/"
                       "test_ELAsTiCC_10000ex_5.0percentTDE.parquet",
    
    # pre-computed BestMall features for the 300-pos training set and test set
    feat_train_csv   = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like/"
                       "training_ELAsTiCC2_sub_300pos_3000neg_BestMallfeat.csv",
    
    feat_train_lab   = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like/"
                       "training_ELAsTiCC2_sub_300pos_3000neg_BestMallfeat_lab.csv",
    feat_train_parq  = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like/"
                       "training_ELAsTiCC2_sub_300pos_3000neg_BestMallfeat.parq",
    
    feat_test_csv    = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like/"
                       "test_ELAsTiCC_10000ex_5per _BestMallfeat.csv",
    feat_test_lab    = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like/"
                       "test_ELAsTiCC_10000ex_5per_BestMallfeat_lab.csv",
    feat_test_parq   = "/ceph/hpc/home/contardog/ELASTICC2/mallorn-like/"
                       "test_ELAsTiCC_10000ex_5per_BestMallfeat.parq",

    # ── pretrained model checkpoints ──────────────────────────────────────────
    model_root       = "/ceph/hpc/home/contardog/MALLORN/SupCon_logs/ELASTICC2_mallornlike",
    # each (fold, adon) maps to:
    #   {model_root}/{fold}/_elasticc2_{kpos}pos_3000neg_tinyermidshallow6_1weight{adon}_checkpoint_val/_bestval
    folds = [
        "supconrun_selfsupsupcon_1-1",
        "supconrun_selfsupsupcon_1-05",
        "supconrun_supcononly",
        "supconrun_selfsuponly",
    ],
    addons           = ["", "_temp007", "_temp007_nounsup"],

    # ── experiment loop ───────────────────────────────────────────────────────
    kpos_list        = [50, 100, 150, 200, 250, 300],

    # ── misc ──────────────────────────────────────────────────────────────────
    dataloader_batch = 512,
    dataloader_workers = 1,
    random_state     = 42,
)



from romae_mallorn.romae_contrastive import RoMAEPreTrainingContrastive, RandomMasking
from romae_mallorn.dataset import MallornDatasetwLabelTrimMask
from romae_mallorn.env_config import MallornConfigContrastiveEnv
from romae_mallorn.utils import override_encoder_size, load_from_checkpoint_contrastive
from romae.model import RoMAEForPreTrainingConfig, EncoderConfig
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 – CLS TOKEN EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(modeldir: str, config) -> RoMAEPreTrainingContrastive:
    """Instantiate and load a checkpoint."""
    encoder_args = override_encoder_size(config.model_size)
    ## Shoulds be useless since eval time only?
    for k, v in dict(
        drop_path_rate=0.15, hidden_drop_rate=0.1, pos_drop_rate=0.1,
        attn_drop_rate=0.1, attn_proj_drop_rate=0.1,
    ).items():
        encoder_args[k] = v

    decoder_size = config.decoder_size or encoder_args["d_model"]
    decoder_args = dict(d_model=decoder_size, nhead=3, depth=2, drop_path_rate=0.05)

    model_config = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=2,
        n_pos_dims=2,
    )
    model = load_from_checkpoint_contrastive(
        modeldir, RoMAEPreTrainingContrastive, RoMAEForPreTrainingConfig, config
    )
    model.set_loss_fn(nn.HuberLoss(reduction="none", delta=0.5))
    return model


def get_cls_tokens(modeldir: str, parqname: str, cfg: dict) -> dict:
    """
    Run the encoder over *parqname* using the checkpoint at *modeldir*.

    Returns
    -------
    dict with keys:
        cls_full   : np.ndarray  [N, d_model]
        cls_contr  : np.ndarray  [N, cls_contrastive_dim]
        labels     : np.ndarray  [N]
        objids     : np.ndarray  [N]
    """
    env_file = modeldir.split("_checkpoint")[0] + "_config.env"
    config = MallornConfigContrastiveEnv(_env_file=env_file)

    model = _build_model(modeldir, config)
    model.eval()

    dataset = MallornDatasetwLabelTrimMask(
        parqname,
        mask_ratio=0.0, gaussian_noise=False,
        obs_dropout_end_trim=0.0, obs_dropout_edge_erosion=0.0,
        gap_threshold_factor=0.0, random_dropout_ratio=0.0,
        training=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["dataloader_batch"],
        num_workers=cfg["dataloader_workers"],
        pin_memory=True,
        collate_fn=torch.utils.data.dataloader.default_collate,
        prefetch_factor=2,
        shuffle=False,
    )

    cls_full_list, cls_contr_list, label_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model.forward_cls(
                batch["values"], batch["mask"],
                batch["positions"], batch["pad_mask"],
                decode=False,
            )
            cls_full_list.append(out["embeddings"][:, 0, :].cpu().numpy())
            cls_contr_list.append(
                out["embeddings"][:, 0, :config.cls_contrastive_dim]
                .cpu().numpy()
            )
            label_list.append(batch["label"].cpu().numpy())
            torch.cuda.empty_cache()

    objids = pl.read_parquet(parqname)["ObjectId"].to_numpy()

    torch.cuda.empty_cache()
    del model, dataset, loader

    return dict(
        cls_full  = np.concatenate(cls_full_list,  axis=0),
        cls_contr = np.concatenate(cls_contr_list, axis=0),
        labels    = np.concatenate(label_list,     axis=0),
        objids    = objids,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 – CLASSIFIERS
# ──────────────────────────────────────────────────────────────────────────────

def kfold_linearsvc_search(
    X, y,
    C_values=(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
    penalties=("l1", "l2"),
    n_splits=5,
    class_weight="balanced",
    scoring="f1",
    random_state=42,
    max_iter=5000,
    verbose=False,          # quiet by default when called in a loop
):
    """
    Stratified k-fold grid-search over LinearSVC penalty + C.
    Returns (results_df, best_config, best_pipeline).
    """
    if hasattr(X, "values"):   # pandas-safe
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for penalty in penalties:
        dual = (penalty == "l2")
        for C in C_values:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", LinearSVC(
                    penalty=penalty, C=C, dual=dual,
                    class_weight=class_weight,
                    max_iter=max_iter, random_state=random_state,
                )),
            ])
            fold_scores = []
            for tr_idx, va_idx in cv.split(X, y):
                pipe.fit(X[tr_idx], y[tr_idx])
                y_pred = pipe.predict(X[va_idx])
                if scoring == "roc_auc":
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y[va_idx], pipe.decision_function(X[va_idx]))
                elif scoring == "f1_macro":
                    score = f1_score(y[va_idx], y_pred, average="macro", zero_division=0)
                else:
                    score = f1_score(y[va_idx], y_pred, average="binary", zero_division=0)
                fold_scores.append(score)

            rows.append(dict(penalty=penalty, C=C,
                             mean_score=np.mean(fold_scores),
                             std_score=np.std(fold_scores)))

    results_df  = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
    best        = results_df.iloc[0]
    best_config = {k: best[k] for k in ("penalty", "C", "mean_score", "std_score")}

    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(
            penalty=best_config["penalty"], C=best_config["C"],
            dual=(best_config["penalty"] == "l2"),
            class_weight=class_weight, max_iter=max_iter, random_state=random_state,
        )),
    ])
    best_model.fit(X, y)

    if verbose:
        print(f"  LinearSVC best → penalty={best_config['penalty']}  "
              f"C={best_config['C']}  {scoring}="
              f"{best_config['mean_score']:.4f}±{best_config['std_score']:.4f}")
        print(classification_report(y, best_model.predict(X), zero_division=0))

    return results_df, best_config, best_model


# ── BestMall ensemble (unchanged from notebook) ───────────────────────────────

MORPHOLOGY_FEATURES = [
    "rest_rise_time","rest_fade_time","rest_fwhm","ls_time",
    "rise_fade_ratio","compactness","rise_slope","flux_kurtosis",
    "robust_duration","duty_cycle","pre_peak_var","amplitude",
]
PHYSICS_FEATURES = [
    "tde_power_law_error","template_chisq_tde","linear_decay_slope",
    "mean_color_gr","std_color_gr","total_radiated_energy","color_monotonicity",
    "negative_flux_fraction","rise_fireball_error","reduced_chi_square",
    "ls_wave","fade_shape_correlation","baseline_ratio","color_cooling_rate",
    "color_slope_gr","flux_ratio_ug","flux_ratio_gr",
    "ug_peak","gr_peak","ur_peak","redshift",
    "absolute_magnitude_proxy","log_tde_error",
]
_MODEL_SEED = 67


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, scale_pos_weight=1.0):
        self.scale_pos_weight = scale_pos_weight
        self.models = {}

    def fit(self, X, y):
        cb_params = dict(
            iterations=1000, depth=5, learning_rate=0.02,
            l2_leaf_reg=10, rsm=0.5, loss_function="Logloss",
            verbose=0, allow_writing_files=False,
            random_seed=_MODEL_SEED,
            scale_pos_weight=self.scale_pos_weight,
        )
        self.models["base"] = CatBoostClassifier(**cb_params)
        self.models["base"].fit(X, y)
        self.feature_importances_ = self.models["base"].feature_importances_

        for name, cols in [("morphology", MORPHOLOGY_FEATURES), ("physics", PHYSICS_FEATURES)]:
            keep = [c for c in cols if c in X.columns]
            if keep:
                self.models[name] = CatBoostClassifier(**cb_params)
                self.models[name].fit(X[keep], y)

        mlp_pipe = Pipeline([
            ("var", VarianceThreshold(threshold=0.0)),
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler()),
            ("net", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                  solver="adam", alpha=0.01, max_iter=600,
                                  random_state=_MODEL_SEED)),
        ])
        self.models["mlp"] = mlp_pipe.fit(X, y)

        knn_pipe = Pipeline([
            ("var", VarianceThreshold(threshold=0.0)),
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", p=2)),
        ])
        self.models["knn"] = knn_pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        p_base  = self.models["base"].predict_proba(X)[:, 1]
        p_morph = (self.models["morphology"].predict_proba(
                       X[[c for c in MORPHOLOGY_FEATURES if c in X.columns]])[:, 1]
                   if "morphology" in self.models else p_base)
        p_phys  = (self.models["physics"].predict_proba(
                       X[[c for c in PHYSICS_FEATURES if c in X.columns]])[:, 1]
                   if "physics" in self.models else p_base)
        p_mlp = self.models["mlp"].predict_proba(X)[:, 1]
        p_knn = self.models["knn"].predict_proba(X)[:, 1]
        final = 0.48*p_base + 0.16*p_morph + 0.16*p_phys + 0.10*p_mlp + 0.10*p_knn
        return np.vstack([1 - final, final]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_with_cv(X, y, use_catboost=True, random_state=67):
    """
    Stratified 5-fold CV with automatic class-weight scaling.
    Returns (final_model, avg_f1, avg_threshold).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores, best_thresholds = [], []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        X_va = X.iloc[va_idx] if hasattr(X, "iloc") else X[va_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
        y_va = y.iloc[va_idx] if hasattr(y, "iloc") else y[va_idx]

        n_pos = y_tr.sum()
        w = (len(y_tr) - n_pos) / n_pos if n_pos > 0 else 1.0
        mdl = EnsembleClassifier(scale_pos_weight=w) if use_catboost else None
        mdl.fit(X_tr, y_tr)

        probs = mdl.predict_proba(X_va)[:, 1]
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.2, 0.8, 0.02):
            s = f1_score(y_va, (probs >= t).astype(int), zero_division=0)
            if s > best_f1:
                best_f1, best_t = s, t
        cv_scores.append(best_f1)
        best_thresholds.append(best_t)

    avg_f1     = float(np.mean(cv_scores))
    avg_thresh = float(np.mean(best_thresholds))

    n_pos_all = y.sum()
    final_w   = (len(y) - n_pos_all) / n_pos_all
    final_mdl = EnsembleClassifier(scale_pos_weight=final_w)
    final_mdl.fit(X, y)
    return final_mdl, avg_f1, avg_thresh


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 – LOGGING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _metric_row(tag: str, kpos: int, labels_true, preds_binary) -> dict:
    """Build a flat dict of common metrics for one (tag, kpos) combo."""
    report = classification_report(labels_true, preds_binary,
                                   output_dict=True, zero_division=0)
    return dict(
        tag         = tag,
        kpos        = kpos,
        f1          = f1_score(labels_true, preds_binary, zero_division=0),
        precision_1 = report.get("1", report.get(1, {})).get("precision", float("nan")),
        recall_1    = report.get("1", report.get(1, {})).get("recall",    float("nan")),
        support_1   = report.get("1", report.get(1, {})).get("support",   0),
        f1_macro    = report["macro avg"]["f1-score"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 – PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_f1_vs_kpos(metrics_df: pd.DataFrame, output_dir: Path):
    """
    One curve per (tag, classifier_type) showing F1 vs number of positives.
    Saves PNG to output_dir.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    tags = metrics_df["tag"].unique()
    cmap = plt.cm.tab20(np.linspace(0, 1, len(tags)))

    for color, tag in zip(cmap, sorted(tags)):
        sub = metrics_df[metrics_df["tag"] == tag].sort_values("kpos")
        ax.plot(sub["kpos"], sub["f1"], marker="o", label=tag, color=color)

    ax.set_xlabel("Number of positive training examples", fontsize=12)
    ax.set_ylabel("F1 ", fontsize=12)
    ax.set_title("Test F1 vs training positives – all configurations", fontsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "f1_vs_kpos.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {path}")


def plot_cls_umap(cls_array: np.ndarray, labels: np.ndarray,
                  title: str, output_dir: Path):
    """
    2D UMAP of CLS tokens coloured by label.  Falls back to PCA if umap
    is not installed.
    """
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        tag = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        tag = "PCA"

    emb = reducer.fit_transform(cls_array)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, marker, color in [(0, ".", "#4c72b0"), (1, "*", "#dd8452")]:
        mask = labels == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, marker=marker,
                   s=8 if lbl == 0 else 40, alpha=0.6,
                   label=f"class {lbl} (n={mask.sum()})")
    ax.set_title(f"{tag} of CLS tokens – {title}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    safe = title.replace("/", "_").replace(" ", "_")
    path = output_dir / f"cls_{tag.lower()}_{safe}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 – MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: dict, output_dir: Path, do_plot: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    cls_dir = output_dir / "cls_tokens"
    cls_dir.mkdir(exist_ok=True)

    # ── load pre-computed BestMall features (300-pos superset) ───────────────
    print("[init] loading BestMall features …")
    feat_all300 = pd.read_csv(cfg["feat_train_csv"])
    lab_all300  = pd.read_csv(cfg["feat_train_lab"])["binary_class"]
    parq_all300 = pl.read_parquet(cfg["feat_train_parq"])

    feat_test     = pd.read_csv(cfg["feat_test_csv"])
    lab_test      = pd.read_csv(cfg["feat_test_lab"])["binary_class"]

    all_metrics: list[dict] = []   # flat rows, written to CSV at the end

    # ── outer loop: number of positives ──────────────────────────────────────
    for kpos in cfg["kpos_list"]:
        print(f"\n{'='*60}")
        print(f"  kpos = {kpos}")
        print(f"{'='*60}")

        parq_kname = (f"{cfg['data_root']}/"
                      f"training_ELAsTiCC2_sub_{kpos}pos_3000neg.parq")
        parq_k = pl.read_parquet(parq_kname).filter(pl.col("binary_class") != -1)

        # sub-select BestMall features that belong to this kpos subset
        mask_kpo    = np.isin(parq_all300["ObjectId"].to_numpy(),
                              parq_k["ObjectId"].to_numpy())
        feat_kpo    = feat_all300[mask_kpo]
        lab_kpo     = lab_all300[mask_kpo]

        # ── BestMall CatBoost baseline ────────────────────────────────────────
        print("[bestmall] training …")
        mdl_bm, f1_bm_cv, thr_bm = train_with_cv(feat_kpo, lab_kpo)
        pred_bm = mdl_bm.predict(feat_test)
        pred_bm_thr = (mdl_bm.predict_proba(feat_test)[:, 1] >= thr_bm).astype(int)

        row = _metric_row("bestmall_catboost", kpos, lab_test, pred_bm_thr)
        row["cv_f1"] = f1_bm_cv
        all_metrics.append(row)
        print(f"  [bestmall] test F1 = {row['f1']:.4f}")

        # ── inner loop: encoder checkpoints ──────────────────────────────────
        for fold in cfg["folds"]:
            for adon in cfg["addons"]:
                tag = f"{fold}{adon}"
                ckpt = (f"{cfg['model_root']}/{fold}/"
                        f"_elasticc2_{kpos}pos_3000neg_"
                        f"tinyermidshallow6_1weight{adon}"
                        f"_checkpoint_val/_bestval")

                if not os.path.isdir(ckpt):
                    print(f"  [skip] checkpoint not found: {ckpt}")
                    continue

                print(f"\n  [{tag}] kpos={kpos}")

                # ── load / cache CLS tokens ──────────────────────────────────
                cls_train_path = cls_dir / f"{tag}_kpos{kpos}_train_full.npy"
                cls_test_path  = cls_dir / f"{tag}_kpos{kpos}_test_full.npy"
                ctr_train_path = cls_dir / f"{tag}_kpos{kpos}_train_contr.npy"
                ctr_test_path  = cls_dir / f"{tag}_kpos{kpos}_test_contr.npy"
                lab_train_path = cls_dir / f"{tag}_kpos{kpos}_labels_train.npy" ## probably useless
                lab_test_path  = cls_dir / f"{tag}_kpos{kpos}_labels_test.npy"
                obj_train_path = cls_dir / f"{tag}_kpos{kpos}_objids_train.npy"
                obj_test_path  = cls_dir / f"{tag}_kpos{kpos}_objids_test.npy"

                if cls_train_path.exists() and cls_test_path.exists():
                    print("    loading cached CLS tokens …")
                    cls_full    = np.load(cls_train_path)
                    cls_fulltest= np.load(cls_test_path)
                    cls_contr   = np.load(ctr_train_path)
                    cls_contrtest = np.load(ctr_test_path)
                    labels_cls  = np.load(lab_train_path)
                    labels_test_cls = np.load(lab_test_path)
                    objids      = np.load(obj_train_path)
                else:
                    print("    extracting CLS tokens (train) …")
                    t0 = time.time()
                    train_out = get_cls_tokens(ckpt, parq_kname, cfg)
                    print(f"    extracting CLS tokens (test) …")
                    test_out  = get_cls_tokens(ckpt, cfg["test_parq"], cfg)
                    print(f"    CLS extraction took {time.time()-t0:.1f}s")

                    cls_full       = train_out["cls_full"]
                    cls_contr      = train_out["cls_contr"]
                    labels_cls     = train_out["labels"]
                    objids         = train_out["objids"]
                    cls_fulltest   = test_out["cls_full"]
                    cls_contrtest  = test_out["cls_contr"]
                    labels_test_cls= test_out["labels"]

                    # cache to disk
                    np.save(cls_train_path, cls_full)
                    np.save(cls_test_path,  cls_fulltest)
                    np.save(ctr_train_path, cls_contr)
                    np.save(ctr_test_path,  cls_contrtest)
                    np.save(lab_train_path, labels_cls)
                    np.save(lab_test_path,  labels_test_cls)
                    np.save(obj_train_path, objids)
                    np.save(obj_test_path,  test_out["objids"])

                # remove unsupervised rows (label == -1)
                sup_mask = labels_cls != -1
                cls_full_sup   = cls_full[sup_mask]
                cls_contr_sup  = cls_contr[sup_mask]
                labels_sup     = labels_cls[sup_mask]

                # choose feature space: selfsuponly uses full CLS, others use contrastive sub-space
                use_full = (fold == "supconrun_selfsuponly")
                X_train = cls_full_sup   if use_full else cls_contr_sup
                X_test  = cls_fulltest   if use_full else cls_contrtest

                # ── LinearSVC ─────────────────────────────────────────────────
                _, best_cfg_svc, mdl_svc = kfold_linearsvc_search(
                    X_train, labels_sup, random_state=cfg["random_state"]
                )
                pred_svc = mdl_svc.predict(X_test)
                row_svc  = _metric_row(f"{tag}_linearSVC", kpos,
                                       labels_test_cls, pred_svc)
                row_svc["best_C"]       = best_cfg_svc["C"]
                row_svc["best_penalty"] = best_cfg_svc["penalty"]
                row_svc["cv_f1"]        = best_cfg_svc["mean_score"]
                all_metrics.append(row_svc)
                print(f"    LinearSVC  test F1 = {row_svc['f1']:.4f}")

                # ── CatBoost on CLS ───────────────────────────────────────────
                mdl_cat, f1_cat_cv, thr_cat = train_with_cv(
                    pd.DataFrame(X_train), pd.Series(labels_sup)
                )
                proba_cat = mdl_cat.predict_proba(pd.DataFrame(X_test))[:, 1]
                pred_cat  = (proba_cat >= thr_cat).astype(int)
                row_cat   = _metric_row(f"{tag}_catboost_CLS", kpos,
                                        labels_test_cls, pred_cat)
                row_cat["cv_f1"] = f1_cat_cv
                all_metrics.append(row_cat)
                print(f"    CatBoost   test F1 = {row_cat['f1']:.4f}")

                # ── optional: UMAP/PCA of CLS ─────────────────────────────────
                if do_plot:
                    plot_cls_umap(
                        X_test, labels_test_cls,
                        f"{tag}_kpos{kpos}",
                        output_dir,
                    )

    # ── save aggregated metrics ───────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    csv_path   = output_dir / "all_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n[done] metrics saved → {csv_path}")

    if do_plot:
        plot_f1_vs_kpos(metrics_df, output_dir)

    return metrics_df


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CLS token classification experiment")
    p.add_argument("--output_dir", default="results",
                   help="Directory to write metrics CSV, CLS arrays, and plots")
    p.add_argument("--plot", action="store_true",
                   help="Generate F1 curves")
    p.add_argument("--kpos", nargs="+", type=int,
                   help="Override kpos_list, e.g. --kpos 50 100 300")
    p.add_argument("--folds", nargs="+",
                   help="Override folds list")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = CONFIG.copy()
    if args.kpos:
        cfg["kpos_list"] = args.kpos
    if args.folds:
        cfg["folds"] = args.folds

    metrics = run(cfg, Path(args.output_dir), do_plot=args.plot)
    print("\nFinal summary:")
    print(metrics.to_string(index=False))