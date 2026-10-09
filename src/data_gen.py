"""M1: generate data/drains.csv and data/sensors.csv.  Run: python -m src.data_gen"""
import numpy as np
import pandas as pd
from src.config import *

BASE_RATE = {"market": 0.6, "near_construction": 0.5, "highway": 0.25, "residential": 0.2}
CRIT_BASE = {"market": 4, "highway": 4, "near_construction": 3, "residential": 2}


def make_drains(rng):
    n = N_DRAINS
    lat0, lon0 = CITY_CENTER
    lat = lat0 + rng.uniform(-0.04, 0.04, n)
    lon = lon0 + rng.uniform(-0.04, 0.04, n)
    loc = rng.choice(LOCATION_TYPES, n, p=[0.25, 0.40, 0.20, 0.15])
    elev = 10 + 8 * np.sin((lat - lat0) * 80) + 6 * np.cos((lon - lon0) * 70) + rng.normal(0, 1.5, n)
    elev = np.clip(elev, 1, None).round(1)
    age = rng.integers(1, 31, n)
    past = rng.poisson(pd.Series(loc).map(BASE_RATE).values * 4 * (0.5 + age / 20))
    days_since = rng.integers(5, 240, n)
    crit = np.clip(pd.Series(loc).map(CRIT_BASE).values + (elev < 8).astype(int)
                   + rng.integers(-1, 2, n), 1, 5)
    return pd.DataFrame({
        "drain_id": [f"D{i+1:03d}" for i in range(n)],
        "name": [f"{l.replace('_', ' ').title()} Drain #{i+1}" for i, l in enumerate(loc)],
        "lat": lat.round(5), "lon": lon.round(5), "location_type": loc,
        "elevation_m": elev, "age_years": age,
        "last_cleaned": (pd.Timestamp(SIM_START) - pd.to_timedelta(days_since, unit="D")).normalize(),
        "past_blockages": past, "criticality": crit,
    })


def make_rainfall(rng, T):
    """City-wide rainfall (mm/hr) with gaussian-shaped storms. One storm is forced near
    the end so the 'current' demo moment has interesting activity."""
    rain = np.zeros(T)
    storms = []
    for k in range(int(rng.integers(4, 7))):
        dur = int(rng.integers(2, 7) * STEPS_PER_HOUR)                 # 2-6 h
        start = (T - dur - int(rng.integers(6, 60))) if k == 0 else int(rng.integers(0, T - dur))
        peak = rng.uniform(15, 60)
        x = np.linspace(-2, 2, dur)
        profile = peak * np.exp(-x ** 2) * (1 + 0.3 * rng.standard_normal(dur)).clip(0.3, None)
        rain[start:start + dur] += profile
        storms.append((start, start + dur))
    drizzle = np.where(rng.random(T) < 0.03, rng.uniform(0, 2, T), 0)
    return (rain + drizzle).clip(0), storms


def make_blockage_severity(rng, drains, storms, T):
    """sev[t, j] in [0,1]: 0 = clear, 1 = fully blocked. Episodes ramp up over 1-3 h."""
    N = len(drains)
    sev = np.zeros((T, N))
    dsc = (pd.Timestamp(SIM_START) - pd.to_datetime(drains.last_cleaned)).dt.days.values
    mult = (1 + dsc / 120) * (1 + drains.age_years.values / 40) * (1 + 0.1 * drains.past_blockages.values)
    lam = np.clip(drains.location_type.map(BASE_RATE).values * mult, 0, 4)
    for j in range(N):
        for _ in range(rng.poisson(lam[j])):
            if rng.random() < 0.5:                                      # storm-triggered
                s0, s1 = storms[int(rng.integers(len(storms)))]
                start = int(rng.integers(max(0, s0 - STEPS_PER_HOUR), s1))
            else:
                start = int(rng.integers(0, T - 1))
            end = min(T, start + int(rng.integers(3, 25) * STEPS_PER_HOUR))   # 3-24 h
            final = rng.uniform(0.4, 1.0)
            ramp = int(rng.integers(1, 4) * STEPS_PER_HOUR)
            profile = np.minimum(1, np.arange(end - start) / ramp) * final
            sev[start:end, j] = np.maximum(sev[start:end, j], profile)
    return sev


def simulate_sensors(rng, drains, rain, sev):
    """Simple hydraulics: blockage cuts flow and capacity; the excess accumulates as
    stored water that raises the level. Returns long-format DataFrame."""
    T, N = sev.shape
    catch = rng.uniform(0.6, 1.5, N)        # L/s of inflow per mm/hr of rain
    cap = rng.uniform(60, 100, N)           # free-flow capacity, L/s
    S = np.zeros(N)
    level, flow, turb = (np.zeros((T, N)) for _ in range(3))
    for t in range(T):
        inflow = 5 + catch * rain[t]
        eff_cap = cap * (1 - 0.9 * sev[t])
        f = np.minimum(inflow * (1 - 0.5 * sev[t]), eff_cap)
        S = np.maximum(0, S * 0.97 + (inflow - f) * 0.05)
        level[t] = np.minimum(8 + 1.5 * np.sqrt(f) + S, 120)
        flow[t] = f
        turb[t] = 15 + 0.6 * rain[t] + 60 * sev[t]
    level += rng.normal(0, 0.8, level.shape)
    flow = (flow * (1 + rng.normal(0, 0.05, flow.shape))).clip(0)
    turb += rng.normal(0, 3, turb.shape)
    for arr in (level, flow, turb):                                    # ~1% sensor dropout
        arr[rng.random(arr.shape) < 0.01] = np.nan

    ts = pd.date_range(SIM_START, periods=T, freq=f"{FREQ_MIN}min")
    return pd.DataFrame({
        "timestamp": np.repeat(ts.values, N),
        "drain_id": np.tile(drains.drain_id.values, T),
        "water_level_cm": level.ravel().round(2),
        "flow_rate_lps": flow.ravel().round(2),
        "turbidity_ntu": turb.ravel().round(2),
        "rainfall_mmhr": np.repeat(rain, N).round(2),
        "is_blocked": (sev > 0.25).astype(int).ravel(),                # ground truth (never a feature)
    })


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    T = DAYS * 24 * STEPS_PER_HOUR
    drains = make_drains(rng)
    rain, storms = make_rainfall(rng, T)
    sev = make_blockage_severity(rng, drains, storms, T)
    sensors = simulate_sensors(rng, drains, rain, sev)
    drains.to_csv(DATA_DIR / "drains.csv", index=False)
    sensors.to_csv(DATA_DIR / "sensors.csv", index=False)
    print(f"drains={len(drains)} sensor_rows={len(sensors)} storms={len(storms)} "
          f"blocked_rate={sensors.is_blocked.mean():.1%} "
          f"drains_ever_blocked={sensors.groupby('drain_id').is_blocked.max().sum()}")


if __name__ == "__main__":
    main()