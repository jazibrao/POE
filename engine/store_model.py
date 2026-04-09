import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from .config import EngineConfig

logger = logging.getLogger(__name__)

def _robust_minmax(x: np.ndarray, p_low: float, p_high: float):
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi == lo:
        return np.zeros_like(x), float(lo), float(hi)
    return np.clip((x - lo) / (hi - lo), 0, 1), float(lo), float(hi)

def build_total_target_model(stores: pd.DataFrame, cfg: EngineConfig) -> dict:
    df = stores.copy()

    # ------------------------------------------------------------------
    # Efficient store definition: Hybrid GMROI × ROS (geometric mean)
    # ------------------------------------------------------------------
    # Rationale: A store should be profitable (GMROI) AND move product (ROS)
    # to be considered a benchmark. The geometric mean penalises imbalance —
    # a store that is world-class on one metric but poor on the other will
    # score lower than one that is good on both.
    #
    # Efficiency_Score = GMROI_pctl^w_g × ROS_pctl^w_r
    # where w_g + w_r = 1.0 (default 50/50)
    # ------------------------------------------------------------------
    w_eff_gmroi = float(getattr(cfg, "efficiency_gmroi_weight", 0.50))
    w_eff_ros = float(getattr(cfg, "efficiency_ros_weight", 0.50))

    gmroi_vals = df["GMROI"].astype(float)
    gmroi_pctl = gmroi_vals.rank(pct=True, method="average").fillna(0)
    df["Store_GMROI_Pctl"] = gmroi_pctl

    if w_eff_ros > 0 and "Overall_ROS" in df.columns:
        ros_vals = pd.to_numeric(df["Overall_ROS"], errors="coerce").fillna(0)
        ros_pctl = ros_vals.rank(pct=True, method="average").fillna(0)
        df["Store_ROS_Pctl"] = ros_pctl

        # Geometric mean: (GMROI_pctl^w_g) * (ROS_pctl^w_r)
        # Shift pctl slightly above zero to avoid log(0) in power calculation
        eps = 0.01
        df["Efficiency_Score"] = (
            (gmroi_pctl.clip(lower=eps) ** w_eff_gmroi)
            * (ros_pctl.clip(lower=eps) ** w_eff_ros)
        )
        eff_thr = float(np.percentile(df["Efficiency_Score"].values, 100 * (1 - cfg.efficient_top_pct)))
        df["Is_Efficient"] = (df["Efficiency_Score"] >= eff_thr).astype(int)
        logger.info(
            f"Hybrid efficiency: GMROI weight={w_eff_gmroi}, ROS weight={w_eff_ros}, "
            f"threshold={eff_thr:.4f}, efficient stores={int(df['Is_Efficient'].sum())}"
        )
    else:
        # Fallback: legacy GMROI-only definition
        df["Store_ROS_Pctl"] = 0.0
        df["Efficiency_Score"] = gmroi_pctl
        eff_thr = float(np.percentile(gmroi_vals.values, 100 * (1 - cfg.efficient_top_pct)))
        df["Is_Efficient"] = (gmroi_vals >= eff_thr).astype(int)
        logger.info(f"Legacy GMROI-only efficiency: threshold={eff_thr:.2f}, efficient stores={int(df['Is_Efficient'].sum())}")

    spf_log = np.log1p(df["SPF"].astype(float).values)
    spf_norm, spf_p5, spf_p95 = _robust_minmax(spf_log, cfg.robust_p_low, cfg.robust_p_high)

    q = df["Q_SCORE"].astype(float).values
    q_norm, q_p5, q_p95 = _robust_minmax(q, cfg.robust_p_low, cfg.robust_p_high)

    y = df["Average of SKU_DC"].astype(float).values
    rft_raw = df["RFT"].astype(float).values
    # Audit 2.4: Log-transform RFT to stabilize long-tail capacity effects
    rft = np.log1p(np.clip(rft_raw, 0, None))
    eff_mask = df["Is_Efficient"].values.astype(bool)

    weights = np.round(np.linspace(0, 1, 21), 2)

    # ------------------------------------------------------------------
    # Robust CV handling
    # Some users may run the engine on a small subset of stores or the
    # chosen efficient_top_pct may produce < 5 efficient stores. In that
    # case, fixed 5-fold CV will crash.
    #
    # We dynamically set n_splits = min(5, n_eff), and if n_eff < 2 we
    # fallback to using all stores for training (with CV disabled).
    # ------------------------------------------------------------------
    n_eff = int(eff_mask.sum())
    use_eff_for_training = n_eff >= 2

    if not use_eff_for_training:
        # Fallback: train on all stores (still keeps the idea of "efficient"
        # for reporting, but prevents a hard crash).
        eff_mask = np.ones_like(eff_mask, dtype=bool)
        n_eff = int(eff_mask.sum())

    # Choose n_splits so that each fold has at least ~2 samples for scoring.
    # (R2 is undefined when a test fold has only 1 sample.)
    n_splits = min(5, max(2, n_eff // 2)) if n_eff >= 4 else 2
    # If still degenerate (e.g., only 1 store total), we cannot do CV.
    kf = None if n_eff < 2 else KFold(n_splits=min(n_splits, n_eff), shuffle=True, random_state=42)
    best = {"cv": -1e9, "w": None, "model": None}

    def Xs(gate):
        XA = (rft * gate).reshape(-1, 1)
        XB = np.column_stack([rft, gate])
        XC = np.column_stack([rft, gate, rft * gate])
        return {"ModelA": XA, "ModelB": XB, "ModelC": XC}

    if cfg.auto_select_gate_weight and kf is not None and n_eff >= 4:
        for w in weights:
            gate = w * spf_norm + (1 - w) * q_norm
            for name, X in Xs(gate).items():
                cv = cross_val_score(Ridge(alpha=1.0), X[eff_mask], y[eff_mask], cv=kf, scoring="r2")
                cv_mean = float(np.nanmean(cv))
                if np.isfinite(cv_mean) and cv_mean > best["cv"]:
                    best.update({"cv": cv_mean, "w": float(w), "model": name})
    else:
        # Either CV selection disabled by config, or CV isn't possible.
        # We choose a stable default and mark CV as NaN.
        _den = float(cfg.fixed_gate_weight_spf) + float(getattr(cfg, "fixed_gate_weight_q", 0.0))
        w_fixed = float(cfg.fixed_gate_weight_spf) / (_den if _den else 1.0)
        best.update({"w": w_fixed, "model": "ModelB", "cv": float("nan")})

    # If CV produced no finite score, fallback to fixed weight.
    if best.get("w") is None:
        _den = float(cfg.fixed_gate_weight_spf) + float(getattr(cfg, "fixed_gate_weight_q", 0.0))
        w_fixed = float(cfg.fixed_gate_weight_spf) / (_den if _den else 1.0)
        best.update({"w": w_fixed, "model": "ModelB", "cv": float("nan")})

    w = best["w"]
    gate = w * spf_norm + (1 - w) * q_norm
    X_best = Xs(gate)[best["model"]]
    reg = Ridge(alpha=1.0).fit(X_best[eff_mask], y[eff_mask])

    df["SPF_log"] = spf_log
    df["SPF_norm"] = spf_norm
    df["Q_norm"] = q_norm
    df["Performance_Gate"] = gate
    df["RFT_raw"] = rft_raw
    df["RFT_log"] = rft
    df["RFTxGate"] = rft * gate
    df["Target_SKU_Total"] = np.maximum(0, np.round(reg.predict(X_best))).astype(int)

    return {
        "stores_enriched": df,
        "model": {
            "best_weight_spf": float(w),
            "best_weight_q": float(1.0 - float(w)),
            "best_model": str(best["model"]),
            "cv_r2": float(best["cv"]),
            "train_r2": float(reg.score(X_best[eff_mask], y[eff_mask])),
            "intercept": float(reg.intercept_),
            "coef": [float(c) for c in reg.coef_],
            "efficient_threshold": float(eff_thr),
            "efficiency_method": "hybrid_gmroi_ros" if w_eff_ros > 0 and "Overall_ROS" in df.columns else "gmroi_only",
            "efficiency_gmroi_weight": float(w_eff_gmroi),
            "efficiency_ros_weight": float(w_eff_ros),
            "scaling": {"spf_log_p5": spf_p5, "spf_log_p95": spf_p95, "q_p5": q_p5, "q_p95": q_p95},
        },
    }
