"""Experiment 4: empirical validation against listed TQQQ market quotes.

Reproduces paper Section 13.5: QQQ is the underlying and TQQQ (beta=3,
gamma=3) is the three-times leveraged product. Listed American-style QQQ and
TQQQ option quotes on three market dates (Section 13.5.1) are converted to
European-equivalent Black implied volatilities (paper flagged uncertainty on
the exact conversion method; this repository uses the Cox-Ross-Rubinstein
tree conversion of ``src.american_option``). A Dupire local-volatility
surface is calibrated to the converted QQQ smile at each listed expiry, the
TQQQ forward is backed out from put-call parity on the converted European
prices at the strike where call and put are closest (Section 13.5.1, step 4),
and the LETF smile is produced by the three compared methods: the forward
joint-law PDE reference (Baseline grid, Table 1), the conditional-Gamma
closure ("2ndOrder", two-moment, Section 7), and the first-order strike
remapping (Section 4).

Run: python -m exp.exp4_tqqq_market_validation [--full]
"""
from __future__ import annotations

import argparse
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import MassiveDataError, load_option_chain, spot_price_on
from src.american_option import american_implied_vol
from src.black_scholes import black_price
from src.config import (EXP4_AMERICAN_TREE_STEPS, EXP4_BETA, EXP4_DIVIDEND_YIELD,
                         EXP4_LETF_TICKER, EXP4_MARKET_DATES, EXP4_RISK_FREE_RATE,
                         EXP4_STRIKE_MONEYNESS_BAND, EXP4_TARGET_TENORS_MONTHS,
                         EXP4_UNDERLYING_TICKER, GRID_CONFIGS_PAPER, GRID_CONFIGS_RUN,
                         MOMENT_GRID_PAPER, MOMENT_GRID_RUN)
from src.dupire_surface import LocalVolSurface, fit_svi_slice
from src.market_inputs import DeterministicInputs
from src.pricing_pipeline import conditional_gamma_smile, gamma_coef_of, joint_law_smile, \
    strike_remapping_smile

RESULTS_DIR = "results"


def _year_fraction(as_of_date: str, expiry: str) -> float:
    """ACT/365 year fraction (not specified in the paper; the same convention
    used for the synthetic maturities of experiments 1-3, src.config)."""
    d0 = pd.Timestamp(as_of_date)
    d1 = pd.Timestamp(expiry)
    return max((d1 - d0).days / 365.0, 1.0 / 365.0)


def _convert_otm_smile(quotes: pd.DataFrame, spot: float, T: float, r: float, q: float) -> pd.DataFrame:
    """European-equivalent Black implied volatility of the out-of-the-money
    side of each strike (OTM calls above spot, OTM puts below spot), via the
    American-to-European tree conversion (src.american_option)."""
    rows = []
    for _, row in quotes.iterrows():
        K = float(row["strike"])
        side = row["contract_type"]
        is_otm = (side == "call" and K >= spot) or (side == "put" and K < spot)
        if not is_otm:
            continue
        iv = american_implied_vol(float(row["mid"]), spot, K, T, r, q, side,
                                   n_steps=EXP4_AMERICAN_TREE_STEPS)
        if iv is None or not np.isfinite(iv):
            continue
        rows.append({"strike": K, "contract_type": side, "iv": iv})
    return pd.DataFrame(rows)


def _tqqq_forward_from_parity(quotes: pd.DataFrame, spot: float, T: float, r: float, q: float,
                               D_T: float) -> tuple[float, float]:
    """Back out F_T^L from put-call parity on European-equivalent prices,
    at the strike where call and put are closest (paper Section 13.5.1, step 4).

    Returns (F_T_L, strike_used).
    """
    calls = quotes[quotes["contract_type"] == "call"][["strike", "mid"]].rename(columns={"mid": "call_mid"})
    puts = quotes[quotes["contract_type"] == "put"][["strike", "mid"]].rename(columns={"mid": "put_mid"})
    paired = calls.merge(puts, on="strike", how="inner")
    if paired.empty:
        raise MassiveDataError("no matched call/put strikes for TQQQ put-call parity")

    F_det = spot * np.exp((r - q) * T)
    euro_call, euro_put = [], []
    for _, row in paired.iterrows():
        K = float(row["strike"])
        vol_c = american_implied_vol(float(row["call_mid"]), spot, K, T, r, q, "call",
                                      n_steps=EXP4_AMERICAN_TREE_STEPS)
        vol_p = american_implied_vol(float(row["put_mid"]), spot, K, T, r, q, "put",
                                      n_steps=EXP4_AMERICAN_TREE_STEPS)
        if vol_c is None or vol_p is None:
            euro_call.append(np.nan)
            euro_put.append(np.nan)
            continue
        euro_call.append(black_price(F_det, K, T, vol_c, D_T, "call"))
        euro_put.append(black_price(F_det, K, T, vol_p, D_T, "put"))
    paired["euro_call"] = euro_call
    paired["euro_put"] = euro_put
    paired = paired.dropna(subset=["euro_call", "euro_put"])
    if paired.empty:
        raise MassiveDataError("American-to-European conversion failed for every matched strike")

    paired["parity_gap"] = (paired["euro_call"] - paired["euro_put"]).abs()
    best = paired.loc[paired["parity_gap"].idxmin()]
    K_star = float(best["strike"])
    F_T_L = K_star + float(best["euro_call"] - best["euro_put"]) / D_T
    return F_T_L, K_star


def run(full: bool) -> None:
    grid_configs = GRID_CONFIGS_PAPER if full else GRID_CONFIGS_RUN
    moment_grid = MOMENT_GRID_PAPER if full else MOMENT_GRID_RUN
    baseline_grid = grid_configs["Baseline"]
    gamma_coef = gamma_coef_of(EXP4_BETA)

    rows, errors = [], []
    for market_date in EXP4_MARKET_DATES:
        try:
            spot_qqq = spot_price_on(EXP4_UNDERLYING_TICKER, market_date)
            spot_tqqq = spot_price_on(EXP4_LETF_TICKER, market_date)
        except MassiveDataError as exc:
            errors.append({"market_date": market_date, "stage": "spot", "error": str(exc)})
            print(f"[exp4] {market_date}: spot lookup failed: {exc}")
            continue
        r = EXP4_RISK_FREE_RATE[market_date]
        q_qqq = EXP4_DIVIDEND_YIELD[EXP4_UNDERLYING_TICKER]
        q_tqqq = EXP4_DIVIDEND_YIELD[EXP4_LETF_TICKER]

        for target_months in EXP4_TARGET_TENORS_MONTHS[market_date]:
            try:
                qqq_chain = load_option_chain(EXP4_UNDERLYING_TICKER, market_date, target_months,
                                               EXP4_STRIKE_MONEYNESS_BAND, spot_qqq)
                tqqq_chain = load_option_chain(EXP4_LETF_TICKER, market_date, target_months,
                                                EXP4_STRIKE_MONEYNESS_BAND, spot_tqqq)
            except MassiveDataError as exc:
                errors.append({"market_date": market_date, "tenor_months": target_months,
                               "stage": "load_chain", "error": str(exc)})
                print(f"[exp4] {market_date} {target_months}M: chain load failed: {exc}")
                continue

            T = _year_fraction(market_date, qqq_chain.expiry)
            D_T = float(np.exp(-r * T))

            qqq_smile = _convert_otm_smile(qqq_chain.quotes, spot_qqq, T, r, q_qqq)
            if len(qqq_smile) < 5:
                errors.append({"market_date": market_date, "tenor_months": target_months,
                               "stage": "qqq_smile", "error": f"only {len(qqq_smile)} usable QQQ quotes"})
                continue
            F_det_qqq = spot_qqq * np.exp((r - q_qqq) * T)
            k_qqq = np.log(qqq_smile["strike"].to_numpy() / F_det_qqq)
            w_qqq = qqq_smile["iv"].to_numpy() ** 2 * T
            svi_params = fit_svi_slice(k_qqq, w_qqq)
            surface = LocalVolSurface(index_name="QQQ", maturity_grid=np.array([T]),
                                       svi_param_grid=[svi_params])

            try:
                F_T_L, K_parity = _tqqq_forward_from_parity(tqqq_chain.quotes, spot_tqqq, T, r,
                                                             q_tqqq, D_T)
            except MassiveDataError as exc:
                errors.append({"market_date": market_date, "tenor_months": target_months,
                               "stage": "tqqq_parity", "error": str(exc)})
                print(f"[exp4] {market_date} {target_months}M: TQQQ parity failed: {exc}")
                continue

            tqqq_smile = _convert_otm_smile(tqqq_chain.quotes, spot_tqqq, T, r, q_tqqq)
            tqqq_bid_iv = tqqq_chain.quotes.copy()
            tqqq_bid_iv["iv_bid"] = [
                american_implied_vol(row["bid"], spot_tqqq, row["strike"], T, r, q_tqqq,
                                      row["contract_type"], n_steps=EXP4_AMERICAN_TREE_STEPS)
                for _, row in tqqq_bid_iv.iterrows()
            ]
            tqqq_bid_iv["iv_ask"] = [
                american_implied_vol(row["ask"], spot_tqqq, row["strike"], T, r, q_tqqq,
                                      row["contract_type"], n_steps=EXP4_AMERICAN_TREE_STEPS)
                for _, row in tqqq_bid_iv.iterrows()
            ]

            inputs = DeterministicInputs(index_name="QQQ", T=T, r=r, q=q_qqq, F_T_S=F_det_qqq,
                                          F_T_L=F_T_L, D_T=D_T)

            market_otm = tqqq_bid_iv[
                ((tqqq_bid_iv["contract_type"] == "call") & (tqqq_bid_iv["strike"] >= F_T_L)) |
                ((tqqq_bid_iv["contract_type"] == "put") & (tqqq_bid_iv["strike"] < F_T_L))
            ].dropna(subset=["iv_bid", "iv_ask"]).sort_values("strike")
            if market_otm.empty:
                errors.append({"market_date": market_date, "tenor_months": target_months,
                               "stage": "tqqq_market_smile", "error": "no usable TQQQ bid/ask quotes"})
                continue

            strikes = market_otm["strike"].to_numpy()
            k_grid = np.log(strikes / F_T_L)

            joint_law = joint_law_smile(surface, EXP4_BETA, T, baseline_grid, inputs, k_grid)
            gamma = conditional_gamma_smile(surface, EXP4_BETA, T, moment_grid, inputs, k_grid,
                                             shifted=False, density_cutoff_relative=1e-6,
                                             variance_negligible_relative=1e-10)
            remap = strike_remapping_smile(surface, EXP4_BETA, T, inputs, k_grid)

            for idx in range(len(strikes)):
                row_market = market_otm.iloc[idx]
                bid_iv, ask_iv = row_market["iv_bid"], row_market["iv_ask"]
                lo_band, hi_band = sorted((bid_iv, ask_iv))
                row = {
                    "market_date": market_date, "tenor_months": target_months, "expiry": qqq_chain.expiry,
                    "T_years": T, "strike": float(strikes[idx]), "k": float(k_grid[idx]),
                    "F_T_L_market": F_T_L, "K_parity_strike": K_parity,
                    "market_bid_iv": bid_iv, "market_ask_iv": ask_iv,
                    "joint_law_iv": joint_law["iv"][idx], "gamma_iv": gamma["iv"][idx],
                    "remap_iv": remap["iv"][idx],
                }
                for method in ("joint_law_iv", "gamma_iv", "remap_iv"):
                    v = row[method]
                    row[f"{method}_in_band"] = bool(np.isfinite(v) and lo_band <= v <= hi_band)
                rows.append(row)
            print(f"[exp4] {market_date} {target_months}M (expiry {qqq_chain.expiry}, T={T:.3f}y): "
                  f"{len(strikes)} TQQQ strikes, F_T_L={F_T_L:.2f} (parity strike {K_parity:.1f})")

    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail.to_csv(f"{RESULTS_DIR}/exp4_tqqq_detail.csv", index=False)
        summary = (detail.groupby(["market_date", "tenor_months"])
                   .agg(n_strikes=("strike", "count"),
                        joint_law_in_band_frac=("joint_law_iv_in_band", "mean"),
                        gamma_in_band_frac=("gamma_iv_in_band", "mean"),
                        remap_in_band_frac=("remap_iv_in_band", "mean"),
                        joint_law_vs_gamma_max_diff_bp=("joint_law_iv", lambda s: np.nan))
                   .reset_index())
        # Joint-law vs conditional-Gamma separation (should be far smaller than the bid-ask spread).
        detail["joint_vs_gamma_abs_diff"] = (detail["joint_law_iv"] - detail["gamma_iv"]).abs()
        detail["bid_ask_spread"] = (detail["market_ask_iv"] - detail["market_bid_iv"]).abs()
        sep = (detail.groupby(["market_date", "tenor_months"])
               .agg(mean_joint_vs_gamma_abs_diff_bp=("joint_vs_gamma_abs_diff", lambda s: s.mean() * 1e4),
                    mean_bid_ask_spread_bp=("bid_ask_spread", lambda s: s.mean() * 1e4))
               .reset_index())
        summary = summary.drop(columns=["joint_law_vs_gamma_max_diff_bp"]).merge(
            sep, on=["market_date", "tenor_months"])
        summary.to_csv(f"{RESULTS_DIR}/exp4_summary.csv", index=False)
        print(summary.to_string(index=False))
        _plot_examples(detail)
    else:
        print("[exp4] no usable market data for any (market_date, tenor); see results/exp4_errors.csv")

    pd.DataFrame(errors).to_csv(f"{RESULTS_DIR}/exp4_errors.csv", index=False)


def _plot_examples(detail: pd.DataFrame) -> None:
    for market_date, tenor in [(EXP4_MARKET_DATES[0], 1), (EXP4_MARKET_DATES[-1], 1)]:
        sub = detail[(detail["market_date"] == market_date) &
                     (detail["tenor_months"] == tenor)].sort_values("k")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.fill_between(sub["k"], sub["market_bid_iv"] * 100, sub["market_ask_iv"] * 100,
                         color="grey", alpha=0.3, label="TQQQ bid-ask band")
        ax.plot(sub["k"], sub["joint_law_iv"] * 100, label="Joint-law PDE (2dPDE)", color="k")
        ax.plot(sub["k"], sub["gamma_iv"] * 100, label="Conditional-Gamma (2ndOrder)", color="C0",
                linestyle="--")
        ax.plot(sub["k"], sub["remap_iv"] * 100, label="First-order remapping (1stOrder)", color="C3",
                linestyle=":")
        ax.set_xlabel("LETF forward log-moneyness k = log(K/F_T^L)")
        ax.set_ylabel("Black implied volatility (%)")
        ax.set_title(f"TQQQ {market_date} {tenor}M: model curves vs listed market quotes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/exp4_tqqq_{market_date}_{tenor}M.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's Table 1 Baseline grid at full resolution")
    args = parser.parse_args()
    try:
        run(full=args.full)
    except Exception:
        traceback.print_exc()
        raise
