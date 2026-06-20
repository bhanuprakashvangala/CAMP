#!/usr/bin/env python3
"""
Experiment 3 — true round-based active learning.

Setup
-----
Same 80/20 per-bucket split as Exp 1 / Exp 2.  Inside the training pool we
treat c(t) as oracle-known but only revealed when the AL gate picks the row;
this is the standard simulation protocol for AL on a fully-labeled pool.

Round 0:  seed S_0 = uniformly-sampled 5% of training pool, c revealed.
Round k:  fit Joint on S_{k-1} (with IPW sample weights = 1/p_t).
          Score every non-audited training row using the *current* Joint:
            G1 uncertainty: predictive dispersion (sigma in log-space)
            G2 risk:        P(M > bucket capacity)
            G3 novelty:     inverse training count in this bucket
          Combine: 0.4*G1 + 0.4*G2 + 0.2*G3 (matches Naga's weights).
          Select the top-(0.05 * |train|) rows -> reveal their c, append to S_k.

5 rounds total; cumulative budget grows 0.05 -> 0.30 (matches Naga's range).
Compare gate-AL vs random-AL vs full-audit oracle on held-out.

Output: results_active_learning.csv  +  figures/fig_e3_4_al_curves.png
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from scipy.stats import norm
warnings.filterwarnings("ignore")

ROOT  = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "selective_audit"
OUT   = Path(__file__).resolve().parent
EPS, SEED = 1.0, 42
Z95 = 1.6448536269514722
N_ROUNDS = 5
SEED_FRAC = 0.05      # fraction audited at round 0
ROUND_FRAC = 0.05     # fraction audited per AL round (5 rounds * 5% = 25%, +5% seed = 30%)

EXP1_FILE = AUDIT / "experiment1" / "budget_0.3_Exp1" / "exp1_lgbm_sizey_scores_b_0.3.csv"
EXP2_FILE = AUDIT / "experiment1" / "experiment2" / "budget_0.1_Exp2" / "budget_0.3_Exp2" / "exp2_qlgbm_sizey_scores_b_0.3.csv"


def split_80_20(df):
    rng = np.random.RandomState(SEED)
    df = df.copy()
    df["partition"] = "train"
    for (wf, proc), grp in df.groupby(["workflow", "process"], sort=False):
        idx = grp.index.values.copy(); rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        df.loc[idx[cut:], "partition"] = "test"
    return df


def fit_joint_point(sub, weights=None):
    X = np.column_stack([np.log(sub["a_MB"].values + EPS),
                          np.log(sub["c_MB"].values + EPS)])
    y = np.log(sub["M_MB"].values + EPS)
    m = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                       verbose=-1, random_state=SEED)
    m.fit(X, y, sample_weight=weights)
    return m


def fit_joint_quantile(sub, q, weights=None):
    X = np.column_stack([np.log(sub["a_MB"].values + EPS),
                          np.log(sub["c_MB"].values + EPS)])
    y = np.log(sub["M_MB"].values + EPS)
    m = LGBMRegressor(objective="quantile", alpha=q,
                       n_estimators=200, num_leaves=31, learning_rate=0.05,
                       verbose=-1, random_state=SEED)
    m.fit(X, y, sample_weight=weights)
    return m


def predict_log(model, df):
    X = np.column_stack([np.log(df["a_MB"].values + EPS),
                          np.log(df["c_MB"].values + EPS)])
    return model.predict(X)


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
        "MAPE_med": float(np.median(ape) * 100),
        "R2_log":   1.0 - ss_res / ss_tot,
        "Cov2x":    float(((pred / M >= 0.5) & (pred / M <= 2.0)).mean() * 100),
        "Wastage_MB": float(np.maximum(safe - M, 0.0).sum()),
        "OOM_count": int((safe < M).sum()),
        "n_eval":    int(len(M)),
    }


# ---------------- Exp 1 (point) ----------------

def gate_scores_exp1(model_pair, candidates, train_audited):
    """
    Score `candidates` rows for AL acquisition using:
      G1 uncertainty: spread between two boosted models with different seeds (bagged var)
      G2 risk: pred / bucket capacity (90th percentile of audited memory)
      G3 novelty: 1 / sqrt(1 + train_audited_count_in_bucket)
    Returns a combined score per row.
    """
    m1, m2 = model_pair
    X = np.column_stack([np.log(candidates["a_MB"].values + EPS),
                          np.log(candidates["c_MB"].values + EPS)])
    p1 = m1.predict(X); p2 = m2.predict(X)
    pred_log = 0.5 * (p1 + p2)
    pred = np.exp(pred_log)
    sigma = np.abs(p1 - p2) / 2.0  # log-space dispersion proxy
    # G1: normalised log-space spread
    g1 = sigma / (np.percentile(sigma, 95) + 1e-9)
    g1 = np.clip(g1, 0, 1)
    # bucket capacity: per-bucket 90th-pct of audited M
    bucket_cap = train_audited.groupby(["workflow", "process"])["M_MB"].quantile(0.90)
    keys = list(zip(candidates["workflow"].values, candidates["process"].values))
    cap = np.array([bucket_cap.get(k, np.percentile(train_audited["M_MB"], 90))
                     for k in keys])
    # G2: pred / capacity (risk of OOM)
    g2 = np.clip(pred / np.maximum(cap, 1.0), 0, 1)
    # G3: novelty - inverse training count
    bucket_n = train_audited.groupby(["workflow", "process"]).size()
    cnts = np.array([bucket_n.get(k, 0) for k in keys])
    g3 = 1.0 / np.sqrt(1.0 + cnts)
    g3 = g3 / (g3.max() + 1e-9)
    return 0.4 * g1 + 0.4 * g2 + 0.2 * g3


def fit_pair_exp1(sub, weights=None):
    """Two LGBMs with different seeds for variance proxy."""
    m1 = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                       verbose=-1, random_state=SEED)
    m2 = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                       subsample=0.7, subsample_freq=1, verbose=-1, random_state=SEED+7)
    X = np.column_stack([np.log(sub["a_MB"].values + EPS),
                          np.log(sub["c_MB"].values + EPS)])
    y = np.log(sub["M_MB"].values + EPS)
    m1.fit(X, y, sample_weight=weights)
    m2.fit(X, y, sample_weight=weights)
    return (m1, m2)


def eval_pair_exp1(pair, te):
    m1, m2 = pair
    X = np.column_stack([np.log(te["a_MB"].values + EPS),
                          np.log(te["c_MB"].values + EPS)])
    pred_log = 0.5 * (m1.predict(X) + m2.predict(X))
    sigma = np.abs(m1.predict(X) - m2.predict(X)) / 2.0
    pred = np.exp(pred_log)
    # safe = pred * exp(K * residual_std). Use bagged sigma + a small floor.
    safe = np.exp(pred_log + 1.5 * np.maximum(sigma, 0.10))
    return pred, safe


# ---------------- Exp 2 (distribution) ----------------

def gate_scores_exp2(trio, candidates, train_audited):
    """G1 = (q95-q05)/q50, G2 = exceedance prob, G3 = bucket novelty."""
    m05, m50, m95 = trio
    X = np.column_stack([np.log(candidates["a_MB"].values + EPS),
                          np.log(candidates["c_MB"].values + EPS)])
    log_q05 = m05.predict(X); log_q50 = m50.predict(X); log_q95 = m95.predict(X)
    sigma_log = np.clip((log_q95 - log_q05) / (2 * Z95), 0.0, 2.0)
    g1 = sigma_log / (np.percentile(sigma_log, 95) + 1e-9)
    g1 = np.clip(g1, 0, 1)
    bucket_cap = train_audited.groupby(["workflow", "process"])["M_MB"].quantile(0.90)
    keys = list(zip(candidates["workflow"].values, candidates["process"].values))
    cap = np.array([bucket_cap.get(k, np.percentile(train_audited["M_MB"], 90))
                     for k in keys])
    log_cap = np.log(cap + EPS)
    # P(M > cap) under LogNormal(mu_log = log_q50, sigma_log)
    z = (log_cap - log_q50) / np.maximum(sigma_log, 1e-3)
    g2 = 1.0 - norm.cdf(z)
    g2 = np.clip(g2, 0, 1)
    bucket_n = train_audited.groupby(["workflow", "process"]).size()
    cnts = np.array([bucket_n.get(k, 0) for k in keys])
    g3 = 1.0 / np.sqrt(1.0 + cnts)
    g3 = g3 / (g3.max() + 1e-9)
    return 0.4 * g1 + 0.4 * g2 + 0.2 * g3


def fit_trio_exp2(sub, weights=None):
    return (fit_joint_quantile(sub, 0.05, weights),
            fit_joint_quantile(sub, 0.50, weights),
            fit_joint_quantile(sub, 0.95, weights))


def eval_trio_exp2(trio, te):
    m05, m50, m95 = trio
    X = np.column_stack([np.log(te["a_MB"].values + EPS),
                          np.log(te["c_MB"].values + EPS)])
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

    # Common seed (so gate / random share the same initial labeled pool)
    rng = np.random.RandomState(SEED)
    seed_idx = rng.choice(tr_all.index.values, size=seed_n, replace=False)

    # Run gate-AL and random-AL in parallel
    gate_audited = tr_all.loc[seed_idx].copy()
    rand_audited = tr_all.loc[seed_idx].copy()
    # Track per-row audit probabilities for IPW (round-by-round)
    gate_pt = np.full(n_tr, fill_value=SEED_FRAC, dtype=float)
    rand_pt = np.full(n_tr, fill_value=SEED_FRAC, dtype=float)

    for k in range(N_ROUNDS + 1):
        cumul_frac = SEED_FRAC + k * ROUND_FRAC if k > 0 else SEED_FRAC

        # --- fit + eval gate strategy ---
        # IPW weight per audited row (clip prob to avoid blowup)
        idx_g = gate_audited.index.values
        wts_g = 1.0 / np.clip(gate_pt[idx_g], 1e-3, 1.0)
        # normalise weights so total equals row count (avoids LGBM scale issues)
        wts_g = wts_g * len(wts_g) / wts_g.sum()
        try:
            model_g = fit_fn(gate_audited, weights=wts_g)
            p_g, s_g = eval_fn(model_g, te)
            mg = metrics(p_g, s_g, te["M_MB"].values)
        except Exception as e:
            print(f"  [gate round {k}] fit failed: {e}")
            mg = {"MAPE_med": np.nan, "R2_log": np.nan, "Cov2x": np.nan,
                   "Wastage_MB": np.nan, "OOM_count": -1, "n_eval": 0}

        # --- fit + eval random strategy (no IPW; uniform) ---
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

        # --- next round's selection (skip after final round) ---
        if k == N_ROUNDS: break
        unaudited_g = tr_all.drop(gate_audited.index)
        unaudited_r = tr_all.drop(rand_audited.index)
        # gate selects top-round_n by score
        scores_g = score_fn(model_g, unaudited_g, gate_audited)
        order = np.argsort(scores_g)[::-1]
        pick_g = unaudited_g.index.values[order[:round_n]]
        # logistic prob from score (calibrated so mean ~= ROUND_FRAC of total)
        # Practical surrogate: p = sigmoid((s - tau)/T), tau at top-round_n threshold
        thr = float(np.partition(scores_g, -round_n)[-round_n]) if len(scores_g) >= round_n else 0
        T = max(0.02, scores_g.std())
        p_round = 1.0 / (1.0 + np.exp(-(scores_g - thr) / T))
        # Update marginal audit probability for IPW (1 - prod(1-p_k))
        gate_pt[unaudited_g.index.values] = 1.0 - (1.0 - gate_pt[unaudited_g.index.values]) * (1.0 - p_round)
        gate_audited = pd.concat([gate_audited, tr_all.loc[pick_g]])

        # random selects round_n uniformly
        pick_r = rng.choice(unaudited_r.index.values, size=round_n, replace=False)
        rand_audited = pd.concat([rand_audited, tr_all.loc[pick_r]])
        rand_pt[unaudited_r.index.values] = 1.0 - (1.0 - rand_pt[unaudited_r.index.values]) * (1.0 - ROUND_FRAC)

    # Oracle reference
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
    d1 = pd.read_csv(EXP1_FILE)
    d1 = d1.dropna(subset=["a_MB","c_MB","M_MB"]).query("a_MB>=0 and c_MB>0 and M_MB>0").reset_index(drop=True)
    d2 = pd.read_csv(EXP2_FILE)
    d2 = d2.dropna(subset=["a_MB","c_MB","M_MB"]).query("a_MB>=0 and c_MB>0 and M_MB>0").reset_index(drop=True)

    rows = []
    rows += run_al_simulation(d1, "exp1",
                                fit_pair_exp1, gate_scores_exp1, eval_pair_exp1)
    rows += run_al_simulation(d2, "exp2",
                                fit_trio_exp2, gate_scores_exp2, eval_trio_exp2)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "results_active_learning.csv", index=False)
    print(f"\nWrote {OUT/'results_active_learning.csv'}")
    print()
    print(out.to_string(index=False))

    # Plot
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
        oracle = out[(out["experiment"] == exp) & (out["strategy"] == "oracle")].iloc[0]

        for metric, ylabel, ax in [
            ("MAPE_med",   "Median APE (%)",     axes[0, col]),
            ("Wastage_MB", "Wastage (MB-units)", axes[1, col]),
        ]:
            for strat in ["random", "gate"]:
                sub = d[d["strategy"] == strat].sort_values("round")
                ax.plot(sub["cumul_audit_frac"], sub[metric],
                          color=color[strat], marker=marker[strat],
                          label=strat.capitalize() + " AL")
            ax.axhline(oracle[metric], color=color["oracle"], linestyle="--",
                          linewidth=2.5, label="Full audit (B=1)")
            ax.set_xlabel("Cumulative audit fraction")
            ax.set_ylabel(ylabel)
            title_p = "Exp 1 (point)" if exp == "exp1" else "Exp 2 (distribution)"
            ax.set_title(f"{title_p} — {ylabel}")
            ax.set_xlim(-0.02, 0.32)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Experiment 3 — Active-learning convergence vs random sampling",
                  fontsize=20, y=1.02)
    fp = OUT / "figures" / "fig_e3_4_al_curves.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"\nWrote {fp}")
    plt.close(fig)


if __name__ == "__main__":
    main()
