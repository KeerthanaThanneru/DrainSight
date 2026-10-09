"""Stateful live simulator: continues the dataset forward, 10 simulated minutes per step.
Same hydraulics as data_gen (blockage cuts flow, excess water stacks up as level)."""
import copy
import numpy as np
import pandas as pd
from src.config import *

BLOCK_BASE = {"market": 0.6, "near_construction": 0.5, "highway": 0.25, "residential": 0.2}
KEEP_HOURS = 48


class LiveSim:
    def __init__(self, sensors: pd.DataFrame, drains: pd.DataFrame, seed: int = SEED + 7):
        self.drains = drains.reset_index(drop=True)
        self.ids = self.drains["drain_id"].values
        N = len(self.ids)
        self.rng = np.random.default_rng(seed)
        r0 = np.random.default_rng(SEED + 1)                       # per-drain hydraulics (approximates data_gen's)
        self.catch, self.cap = r0.uniform(0.6, 1.5, N), r0.uniform(60, 100, N)
        self.prop = self.drains["location_type"].map(BLOCK_BASE).values * (1 + self.drains["age_years"].values / 40)

        keep = sensors[sensors["timestamp"] > sensors["timestamp"].max() - pd.Timedelta(hours=KEEP_HOURS)]
        self.history = keep.sort_values(["timestamp", "drain_id"]).reset_index(drop=True)
        self.now = self.history["timestamp"].max()
        last = self.history[self.history["timestamp"] == self.now].set_index("drain_id").reindex(self.ids)
        self.sev = np.where(last["is_blocked"].fillna(0).values.astype(bool), 0.7, 0.0)   # current severity per drain
        self.target = self.sev.copy()
        lvl, flow = last["water_level_cm"].fillna(10).values, last["flow_rate_lps"].fillna(10).values
        self.S = np.maximum(0, lvl - (8 + 1.5 * np.sqrt(flow)))     # stored water inferred from the last reading
        self.tick = 0                                               # bumps on every state change (cache key)

    # ---- demo controls -------------------------------------------------
    def force_block(self, drain_id: str, severity: float = 0.9):
        j = int(np.where(self.ids == drain_id)[0][0])
        self.target[j] = severity
        self.sev[j] = max(self.sev[j], severity * 0.5)
        self.tick += 1

    def clean_drain(self, drain_id: str):
        j = int(np.where(self.ids == drain_id)[0][0])
        self.sev[j] = self.target[j] = 0.0
        self.S[j] *= 0.2
        self.tick += 1

    def fork(self) -> "LiveSim":
        """Independent copy (same RNG state) for what-if scenarios."""
        return copy.deepcopy(self)

    def rain_series(self) -> pd.Series:
        return self.history.groupby("timestamp")["rainfall_mmhr"].first().sort_index()

    # ---- dynamics ------------------------------------------------------
    def _update_blockages(self, rain: float):
        N = len(self.ids)
        p = self.prop * (0.00005 + 0.0003 * rain)                   # debris washed in: more rain, more new blockages
        u_new, u_sev = self.rng.random(N), self.rng.uniform(0.5, 1.0, N)   # always drawn -> scenarios stay comparable
        new = (self.sev < 0.05) & (self.target < 0.05) & (u_new < p)
        self.target[new] = u_sev[new]
        ramp = 12.0                                                 # ~2 h to reach full severity
        self.sev = np.where(self.target > self.sev, np.minimum(self.target, self.sev + self.target / ramp), self.sev)

    def step(self, rain_mmhr: float, n: int = 1) -> pd.DataFrame:
        N, chunks = len(self.ids), []
        for _ in range(n):
            self.now = self.now + pd.Timedelta(minutes=FREQ_MIN)
            rain = max(0.0, rain_mmhr * (1 + 0.1 * self.rng.standard_normal()))
            self._update_blockages(rain)
            inflow = 5 + self.catch * rain
            f = np.minimum(inflow * (1 - 0.5 * self.sev), self.cap * (1 - 0.9 * self.sev))
            self.S = np.maximum(0, self.S * 0.97 + (inflow - f) * 0.05)
            level = np.minimum(8 + 1.5 * np.sqrt(f) + self.S, 120) + self.rng.normal(0, 0.8, N)
            flow = (f * (1 + self.rng.normal(0, 0.05, N))).clip(0)
            turb = 15 + 0.6 * rain + 60 * self.sev + self.rng.normal(0, 3, N)
            chunks.append(pd.DataFrame({
                "timestamp": self.now, "drain_id": self.ids, "water_level_cm": level.round(2),
                "flow_rate_lps": flow.round(2), "turbidity_ntu": turb.round(2), "rainfall_mmhr": round(rain, 2),
                "is_blocked": (self.sev > 0.25).astype(int)}))
        new = pd.concat(chunks, ignore_index=True)
        self.history = pd.concat([self.history, new], ignore_index=True)
        self.history = self.history[self.history["timestamp"] > self.now - pd.Timedelta(hours=KEEP_HOURS)].reset_index(drop=True)
        self.tick += 1
        return new