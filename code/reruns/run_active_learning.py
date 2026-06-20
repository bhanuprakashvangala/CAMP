#!/usr/bin/env python3
"""
reruns/run_active_learning.py — round-based AL with paper-faithful gates.

Differences from experiment_3/run_active_learning.py
----------------------------------------------------
* Exp 1 gate now consumes NGBoost (LogNormal head) native (mu, sigma).
  - G1 uncertainty:  sigma_log normalised by its 95th percentile.
  - G2 risk:         1 - Phi( (log_cap - mu_log) / sigma_log ).
  - G3 novelty:      unchanged 1 / sqrt(1 + bucket_count) on audited pool.
  - composite:       0.4*G1 + 0.4*G2 + 0.2*G3  (same weights).

* Exp 2 gate is unchanged: Q-LightGBM at alpha in {0.05, 0.50, 0.95},
  sigma_log derived from the quantile spread (q95 - q05) / (2 * Z95).

IPW retraining, round count, seed/round fractions, and oracle reference
are identical to the original.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from scipy.stats import norm
from ngboost import NGBRegressor
from ngboost.distns import LogNormal
warnings.filterwarnings("ignore")

ROOT  = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "selective_audit"
OUT   = Path(__file__).resolve().parent
(OUT / "figures").mkdir(exist_ok=True)
EPS, SEED = 1.0, 42
Z95 = 1.6448536269514722
N_ROUNDS = 5
SEED_FRAC  = 0.05
ROUND_FRAC = 0.05

EXP1_FILE = AUDIT / "experiment1" / "budget_0.3_Exp1" / "exp1_lgbm_sizey_scores_b_0.3.csv"
EXP2_FILE = AUDIT / "experiment1" / "experiment2" / "budget_0.1_Exp2" / "budget_0.3_Exp2" / "exp2_qlgbm_sizey_scores_b_0.3.csv"


# -------------------- shared --------------------
def split_80_20(df):
    rng = np.random.RandomState(SEED)
    df = df.copy()
    df["partition"] = "train"
    for (wf, proc), grp in df.groupby(["workflow", "process"], sort=False):
        idx = grp.index.values.copy(); rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        df.loc[idx[cut:], "partition"] = "test"
    return df


def metrics(pred_mb, safe_mb, M_mb):
    pred = np.asarray(pred_mb, float); safe = np.asarray(safe_mb, float)
    M    = np.asarray(M_mb, float)
    mask = (M > 0) & np.isfinite(pred) & np.isfinite(M)
    pred, safe, M = pred[mask], safe[mask], M[mask]
    ape = np.abs(pred - M) / M
    log_y = np.log(M + EPS); log_p = np.log(np.clip(pred, EPS, None))
    ss_res = float(np.sum((log_y - log_p) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2)) or 1.0
    return {
        "MAPE_med":   float(np.median(ape) * 100),
        "R2_log":     1.0 - ss_res / ss_tot,
        "Cov2x":      float(((pred / M >= 0.5) & (pred / M <= 2.0)).mean() * 100),
        "Wastage_MB": float(np.maximum(safe - M, 0.0).sum()),
        "OOM_count":  int((safe < M).sum()),
        "n_eval":     int(len(M)),
    }


def joint_X(df):
    return np.column_stack([np.log(df["a_MB"].values + EPS),
                            np.log(df["c_MB"].values + EPS)])


# ---------------- Exp 1 (NGBoost LogNormal driver) ----------------

def fit_ngb_exp1(sub, weights=None):
    """Fit NGBoost(LogNormal) on Joint feature view."""
    X = joint_X(sub)
    y_raw = sub["M_MB"].values + EPS  # positive raw target for LogNormal
    m = NGBRegressor(Dist=LogNormal, n_estimators=200, learning_rate=0.05,
                     verbose=False, random_state=SEED)
    m.fit(X, y_raw, sample_weight=weights)
    return m


def ngb_dist(model, df):
    """Return (mu_log, sigma_log) per row from NGBoost LogNormal.

    sigma_log is clipped to [0.05, 2.0]; NGBoost can emit pathologically
    large per-row scales when trained on small/IPW-reweighted subsets.
    The cap mirrors the Exp 2 quantile-spread path."""
    dist = model.pred_dist(joint_X(df))
    mu_log    = np.asarray(dist.loc,   dtype=float)
    sigma_log = np.clip(np.asarray(dist.scale, dtype=float), 0.05, 2.0)
    return mu_log, sigma_log


def gate_scores_exp1(model, candidates, train_audited):
    """Native NGBoost LogNormal driver:
       G1 = sigma_log / pct95(sigma_log)
       G2 = P(M > bucket_cap) under LogNormal(mu_log, sigma_log)
       G3 = 1 / sqrt(1 + bucket_count_in_audited_pool)
    """
    mu_log, sigma_log = ngb_dist(model, candidates)
    # G1 — uncertainty
    pct95 = np.percentile(sigma_log, 95) + 1e-9
    g1 = np.clip(sigma_log / pct95, 0, 1)
    # bucket capacity from audited pool (90th percentile of M)
    bucket_cap = train_audited.groupby(["workflow", "process"])["M_MB"].quantile(0.90)
    keys = list(zip(candidates["workflow"].values, candidates["process"].values))
    cap = np.array([bucket_cap.get(k, np.percentile(train_audited["M_MB"], 90))
                    for k in keys])
    log_cap = np.log(cap + EPS)
    # G2 — risk: P(M > cap) under LogNormal(mu_log, sigma_log)
    z = (log_cap - mu_log) / np.maximum(sigma_log, 1e-3)
    g2 = np.clip(1.0 - norm.cdf(z), 0, 1)
    # G3 — novelty
    bucket_n = train_audited.groupby(["workflow", "process"]).size()
    cnts = np.array([bucket_n.get(k, 0) for k in keys])
    g3 = 1.0 / np.sqrt(1.0 + cnts)
    g3 = g3 / (g3.max() + 1e-9)
    return 0.4 * g1 + 0.4 * g2 + 0.2 * g3


def eval_ngb_exp1(model, te):
    mu_log, sigma_log = ngb_dist(model, te)
    pred = np.exp(mu_log)
    # safe allocation: 1.5 * sigma cushion in log space. sigma_log is the
    # per-row dispersion from NGBoost; ngb_dist already clips it to [0.05, 2.0]
    # to keep the safe bound numerically stable.
    safe = np.exp(mu_log + 1.5 * sigma_log)
    return pred, safe


# ---------------- Exp 2 (Q-LGBM trio — UNCHANGED) ----------------

def fit_joint_quantile(sub, q, weights=None):
    X = joint_X(sub)
    y = np.log(sub["M_MB"].values + EPS)
    m = LGBMRegressor(objective="quantile", alpha=q,
                      n_estimators=200, num_leaves=31, learning_rate=0.05,
                      verbose=-1, random_state=SEED)
    m.fit(X, y, sample_weight=weights)
    return m


def fit_trio_exp2(sub, weights=None):
    return (fit_joint_quantile(sub, 0.05, weights),
            fit_joint_quantile(sub, 0.50, weights),
            fit_joint_quantile(sub, 0.95, weights))


def gate_scores_exp2(trio, candidates, train_audited):
    """Exp 2 distribution route is Q-LGBM with quantile-spread-derived sigma.
    G1 = sigma_log / pct95(sigma_log), G2 = P(M > cap), G3 = bucket novelty."""
    m05, m50, m95 = trio
    X = joint_X(candidates)
    log_q05 = m05.predict(X); log_q50 = m50.predict(X); log_q95 = m95.predict(X)
    sigma_log = np.clip((log_q95 - log_q05) / (2 * Z95), 0.0, 2.0)
    g1 = sigma_log / (np.percentile(sigma_log, 95) + 1e-9)
    g1 = np.clip(g1, 0, 1)
    bucket_cap = train_audited.groupby(["workflow", "process"])["M_MB"].quantile(0.90)
    keys = list(zip(candidates["workflow"].values, candidates["process"].values))
    cap = np.array([bucket_cap.get(k, np.percentile(train_audited["M_MB"], 90))
                    for k in keys])
    log_cap = np.log(cap + EPS)
    z = (log_cap - log_q50) / np.maximum(sigma_log, 1e-3)
    g2 = np.clip(1.0 - norm.cdf(z), 0, 1)
    bucket_n = train_audited.groupby(["workflow", "process"]).size()
    cnts = np.array([bucket_n.get(k, 0) for k in keys])
    g3 = 1.0 / np.sqrt(1.0 + cnts)
    g3 = g3 / (g3.max() + 1e-9)
    return 0.4 * g1 + 0.4 * g2 + 0.2 * g3


def eval_trio_exp2(trio, te):
    m05, m50, m95 = trio
    X = joint_X(te)
    pred = np.exp(m50.predict(X))
    safe = np.exp(m95.predict(X))
    return pred, safe


# ---------------- AL loop ----------------

def run_al_simulation(df_full, exp_tag, fit_fn, score_fn, eval_fn) -> list[dict]:
    df = split_80_20(df_full)
    tr_all = df.query("partition == 'train'").reset_index(drop=True)
    te     = df.query("partition == 'test'").reset_index(drop=True)
    n_tr = len(tr_all); n_te = len(te)
    seed_n  = max(50, int(round(SEED_FRAC * n_tr)))
    round_n = max(50, int(round(ROUND_FRAC * n_tr)))
    print(f"\n=== {exp_tag.upper()}: train={n_tr:,}  test={n_te:,}  "
          f"seed={seed_n}  per-round={round_n} ===")

    rows = []
    rng = np.random.RandomState(SEED)
    seed_idx = rng.choice(tr_all.index.values, size=seed_n, replace=False)

    gate_audited = tr_all.loc[seed_idx].copy()
    rand_audited = tr_all.loc[seed_idx].copy()
    gate_pt = np.full(n_tr, fill_value=SEED_FRAC, dtype=float)
    rand_pt = np.full(n_tr, fill_value=SEED_FRAC, dtype=float)

    for k in range(N_ROUNDS + 1):
        cumul_frac = SEED_FRAC + k * ROUND_FRAC if k > 0 else SEED_FRAC

        idx_g = gate_audited.index.values
        wts_g = 1.0 / np.clip(gate_pt[idx_g], 1e-3, 1.0)
        wts_g = wts_g * len(wts_g) / wts_g.sum()
        try:
            model_g = fit_fn(gate_audited, weights=wts_g)
            p_g, s_g = eval_fn(model_g, te)
            mg = metrics(p_g, s_g, te["M_MB"].values)
        except Exception as e:
            print(f"  [gate round {k}] fit failed: {e}")
            mg = {"MAPE_med": np.nan, "R2_log": np.nan, "Cov2x": np.nan,
                  "Wastage_MB": np.nan, "OOM_count": -1, "n_eval": 0}

        try:
            model_r = fit_fn(rand_audited)
            p_r, s_r = eval_fn(model_r, te)
            mr = metrics(p_r, s_r, te["M_MB"].values)
        except Exception as e:
            print(f"  [random round {k}] fit failed: {e}")
            mr = {"MAPE_med": np.nan, "R2_log": np.nan, "Cov2x": np.nan,
                  "Wastage_MB": np.nan, "OOM_count": -1, "n_eval": 0}

        rows.append({"experiment": exp_tag, "round": k,
                     "cumul_audit_frac": cumul_frac,
                     "n_audited": len(gate_audited),
                     "strategy": "gate", **mg})
        rows.append({"experiment": exp_tag, "round": k,
                     "cumul_audit_frac": cumul_frac,
                     "n_audited": len(rand_audited),
                     "strategy": "random", **mr})

        print(f"  round {k}  cumul={cumul_frac:.2f}  "
              f"gate MAPE={mg['MAPE_med']:.2f} W={mg['Wastage_MB']:.0f} "
              f"OOM={mg['OOM_count']}  |  "
              f"rand MAPE={mr['MAPE_med']:.2f} W={mr['Wastage_MB']:.0f} "
              f"OOM={mr['OOM_count']}")

        if k == N_ROUNDS:
            break

        unaudited_g = tr_all.drop(gate_audited.index)
        unaudited_r = tr_all.drop(rand_audited.index)

        scores_g = score_fn(model_g, unaudited_g, gate_audited)
        order = np.argsort(scores_g)[::-1]
        pick_g = unaudited_g.index.values[order[:round_n]]
        thr = float(np.partition(scores_g, -round_n)[-round_n]) if len(scores_g) >= round_n else 0
        T = max(0.02, scores_g.std())
        p_round = 1.0 / (1.0 + np.exp(-(scores_g - thr) / T))
        gate_pt[unaudited_g.index.values] = (
            1.0 - (1.0 - gate_pt[unaudited_g.index.values]) * (1.0 - p_round)
        )
        gate_audited = pd.concat([gate_audited, tr_all.loc[pick_g]])

        pick_r = rng.choice(unaudited_r.index.values, size=round_n, replace=False)
        rand_audited = pd.concat([rand_audited, tr_all.loc[pick_r]])
        rand_pt[unaudited_r.index.values] = (
            1.0 - (1.0 - rand_pt[unaudited_r.index.values]) * (1.0 - ROUND_FRAC)
        )

    try:
        model_o = fit_fn(tr_all)
        p_o, s_o = eval_fn(model_o, te)
        mo = metrics(p_o, s_o, te["M_MB"].values)
        rows.append({"experiment": exp_tag, "round": -1,
                     "cumul_audit_frac": 1.0,
                     "n_audited": len(tr_all),
                     "strategy": "oracle", **mo})
        print(f"  ORACLE   cumul=1.00  MAPE={mo['MAPE_med']:.2f} "
              f"W={mo['Wastage_MB']:.0f} OOM={mo['OOM_count']}")
    except Exception as e:
        print(f"  oracle fit failed: {e}")
    return rows


def main():
    print("Loading audit drops...")
    d1 = (pd.read_csv(EXP1_FILE)
            .dropna(subset=["a_MB", "c_MB", "M_MB"])
            .query("a_MB >= 0 and c_MB > 0 and M_MB > 0").reset_index(drop=True))
    d2 = (pd.read_csv(EXP2_FILE)
            .dropna(subset=["a_MB", "c_MB", "M_MB"])
            .query("a_MB >= 0 and c_MB > 0 and M_MB > 0").reset_index(drop=True))

    rows = []
    rows += run_al_simulation(d1, "exp1",
                              fit_ngb_exp1,    gate_scores_exp1, eval_ngb_exp1)
    rows += run_al_simulation(d2, "exp2",
                              fit_trio_exp2,   gate_scores_exp2, eval_trio_exp2)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "results_active_learning.csv", index=False)
    print(f"\nWrote {OUT / 'results_active_learning.csv'}")
    print(out.to_string(index=False))

    # Plot (same layout as the original)
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18,
                         "axes.labelsize": 16, "legend.fontsize": 14,
                         "lines.linewidth": 3.0, "lines.markersize": 12,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "figure.constrained_layout.use": True})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    color = {"gate": "#d62728", "random": "#1f77b4", "oracle": "#2ca02c"}
    marker = {"gate": "^", "random": "o", "oracle": "*"}

    for col, exp in enumerate(["exp1", "exp2"]):
        d = out[(out["experiment"] == exp) & (out["strategy"].isin(["gate", "random"]))]
        oracle = out[(out["experiment"] == exp) & (out["strategy"] == "oracle")]
        oracle_row = oracle.iloc[0] if len(oracle) else None

        for metric, ylabel, ax in [
            ("MAPE_med",   "Median APE (%)",     axes[0, col]),
            ("Wastage_MB", "Wastage (MB-units)", axes[1, col]),
        ]:
            for strat in ["random", "gate"]:
                sub = d[d["strategy"] == strat].sort_values("round")
                ax.plot(sub["cumul_audit_frac"], sub[metric],
                        color=color[strat], marker=marker[strat],
                        label=strat.capitalize() + " AL")
            if oracle_row is not None:
                ax.axhline(oracle_row[metric], color=color["oracle"], linestyle="--",
                           linewidth=2.5, label="Full audit (B=1)")
            ax.set_xlabel("Cumulative audit fraction")
            ax.set_ylabel(ylabel)
            title_p = "Exp 1 (NGBoost LogNormal)" if exp == "exp1" else "Exp 2 (Q-LGBM)"
            ax.set_title(f"{title_p} — {ylabel}")
            ax.set_xlim(-0.02, 0.32)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Reruns — AL convergence (Exp 1: NGBoost / Exp 2: Q-LGBM)",
                 fontsize=20, y=1.02)
    fp = OUT / "figures" / "fig_e3_al_curves.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"\nWrote {fp}")
    plt.close(fig)


if __name__ == "__main__":
    main()
