from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
IMG_DIR = DATA_DIR / "images"

SEED = 42
CITY_CENTER = (19.0760, 72.8777)      # (lat, lon): change to your demo city
N_DRAINS = 60
DAYS = 10
FREQ_MIN = 10                         # minutes per sensor reading
STEPS_PER_HOUR = 60 // FREQ_MIN       # 6
HORIZON_STEPS = STEPS_PER_HOUR        # target: blocked within the next 1 hour
SIM_START = "2026-09-28"

LOCATION_TYPES = ["market", "residential", "near_construction", "highway"]
LOC_COLS = [f"loc_{t}" for t in LOCATION_TYPES]
IMAGE_CLASSES = ["clear", "partial", "blocked"]

BASE_COLS = ["water_level_cm", "flow_rate_lps", "turbidity_ntu", "rainfall_mmhr"]
DYN_COLS = ["level_mean_1h", "level_trend_1h", "flow_mean_1h", "flow_trend_1h",
            "turb_mean_1h", "rain_mean_3h", "level_flow_ratio"]
STATIC_COLS = ["days_since_cleaned", "age_years", "past_blockages"]
FEATURE_COLS = BASE_COLS + DYN_COLS + STATIC_COLS + LOC_COLS

ANOMALY_COLS = ["water_level_cm", "flow_rate_lps", "level_flow_ratio",
                "turbidity_ntu", "rain_mean_3h", "level_trend_1h"]

# human-readable names for explanations: col -> (label, unit)
FRIENDLY = {
    "water_level_cm": ("Water level", "cm"),
    "flow_rate_lps": ("Flow rate", "L/s"),
    "turbidity_ntu": ("Turbidity", "NTU"),
    "rainfall_mmhr": ("Rainfall", "mm/hr"),
    "level_mean_1h": ("Avg water level (1h)", "cm"),
    "level_trend_1h": ("Water-level change (1h)", "cm"),
    "flow_mean_1h": ("Avg flow (1h)", "L/s"),
    "flow_trend_1h": ("Flow change (1h)", "L/s"),
    "turb_mean_1h": ("Avg turbidity (1h)", "NTU"),
    "rain_mean_3h": ("Avg rainfall (3h)", "mm/hr"),
    "level_flow_ratio": ("Level-to-flow ratio", ""),
    "days_since_cleaned": ("Days since cleaned", "days"),
    "age_years": ("Drain age", "yrs"),
    "past_blockages": ("Past blockages", ""),
}