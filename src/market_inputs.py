"""Deterministic forward levels and discount factors (paper Section 2).

The model treats the forward levels F_t^S (underlying index) and F_t^L (LETF)
and the discount factor D_t as deterministic inputs. For the synthetic SPX/NKY
tests of experiments 1-3 these are built from flat continuously-compounded
risk-free rate r and dividend yield q (src.config.RISK_FREE_RATE,
DIVIDEND_YIELD, SPOT_LEVEL): F_T^S = S_0 exp((r-q)T), D_T = exp(-rT). The LETF
forward is built from a nominal initial level L_0 (src.config.LETF_INITIAL_LEVEL)
compounded at the same risk-free rate, since the model's I_T (variance-drag)
adjustment already captures the leverage-specific convexity correction and the
paper gives no separate fund financing spread (flagged uncertainty).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import DIVIDEND_YIELD, LETF_INITIAL_LEVEL, RISK_FREE_RATE, SPOT_LEVEL


@dataclass
class DeterministicInputs:
    index_name: str
    T: float
    r: float
    q: float
    F_T_S: float
    F_T_L: float
    D_T: float


def build_inputs(index_name: str, T: float) -> DeterministicInputs:
    r = RISK_FREE_RATE[index_name]
    q = DIVIDEND_YIELD[index_name]
    S0 = SPOT_LEVEL[index_name]
    F_T_S = S0 * np.exp((r - q) * T)
    D_T = np.exp(-r * T)
    F_T_L = LETF_INITIAL_LEVEL * np.exp(r * T)
    return DeterministicInputs(index_name=index_name, T=T, r=r, q=q, F_T_S=F_T_S, F_T_L=F_T_L, D_T=D_T)
