"""Teaching data and calendar-aligned monthly tables."""
from importlib.resources import files
import re
import numpy as np
import pandas as pd
from studio_core.example_prompt import AIRFARE_FRED_COLUMNS


def simulate_data(seed=422, months=240):
    """Simulate from January 2005. Leave out April 2012 to teach calendar lags."""
    if months < 1:
        raise ValueError("months must be positive")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", periods=months, freq="MS")
    old_fuel, old_fare = 0.0, 0.0
    rows = []
    for date in dates:
        fuel = 0.52 * old_fuel + rng.normal(0, 1.8)
        seasonal = 0.40 * np.sin(2 * np.pi * (date.month - 1) / 12)
        fare = (0.15 + 0.42 * old_fare + 0.18 * old_fuel
                + 0.85 * (old_fuel > 1.5) + seasonal + rng.normal(0, 0.48))
        rows.append((date.strftime("%Y-%m-%d"), fare, fuel))
        old_fare, old_fuel = fare, fuel
    df = pd.DataFrame(rows, columns=["date", "fare_inflation", "fuel_inflation"])
    return df.loc[df.date != "2012-04-01"].reset_index(drop=True).round(6)


def load_demo():
    path = files("studio_core").joinpath("data").joinpath("monthly_airfares.csv")
    with path.open("rb") as f:
        return pd.read_csv(f)


def monthly_frame(data):
    if not isinstance(data, pd.DataFrame) or "date" not in data.columns:
        raise ValueError("Provide a DataFrame with a column named 'date'.")
    if data.empty or data.columns.has_duplicates:
        raise ValueError("The data must have observations and unique column names.")
    index = pd.PeriodIndex(pd.to_datetime(data["date"], errors="raise"), freq="M")
    if index.has_duplicates or index.hasnans:
        raise ValueError("Supply exactly one dated row per observed month.")
    frame = data.drop(columns="date").set_axis(index).sort_index()
    return frame.reindex(pd.period_range(index.min(), index.max(), freq="M", name="month"))


def prepare_airfare_sample(data):
    """Build the two rates for the fixed January 2025 example, from levels or prepared rates."""
    if "date" not in data:
        for name in ("observation_date", "DATE"):
            if name in data:
                data = data.rename(columns={name: "date"})
                break
    frame = monthly_frame(data)
    names = list(AIRFARE_FRED_COLUMNS.values())
    if all(name in frame for name in names):
        rates = frame[names].apply(pd.to_numeric, errors="raise")
    elif all(series_id in frame for series_id in AIRFARE_FRED_COLUMNS):
        levels = frame[list(AIRFARE_FRED_COLUMNS)].apply(pd.to_numeric, errors="raise")
        rates = levels.pct_change(fill_method=None).mul(100).rename(columns=AIRFARE_FRED_COLUMNS)
    else:
        raise ValueError("The January 2025 airfare example needs fare_inflation and fuel_inflation "
                         "or the FRED level columns CUSR0000SETG01 and MJFUELUSGULF. "
                         "Choose Forecast example for simulated data, or use FRED API to import both series.")
    rates = rates.loc["2005-01":"2024-12"].replace([np.inf, -np.inf], np.nan)
    latest = rates.reindex(pd.period_range("2024-10", "2024-12", freq="M"))
    if latest["fare_inflation"].isna().any() or pd.isna(latest["fuel_inflation"].iloc[-1]):
        raise ValueError("The January 2025 forecast needs October–December 2024 airfare changes "
                         "and the December 2024 fuel change. Check the observations and preceding month’s levels.")
    rates.index = rates.index.to_timestamp().strftime("%Y-%m-%d")
    return rates.rename_axis("date").reset_index()


def load_retail_demo():
    """U.S. Census Bureau via FRED, MRTSSM44000USN, Jan 2005 through Dec 2024."""
    path = files("studio_core").joinpath("data", "monthly_retail_sales.csv")
    with path.open("rb") as f:
        return pd.read_csv(f)

def retail_sample(data):
    """Apply the published classroom cutoff even when the uploaded file is longer."""
    from studio_core.example_prompt import RETAIL_ID
    frame = monthly_frame(data)
    if RETAIL_ID not in frame:
        raise ValueError(f"The retail exercise requires {RETAIL_ID}. For another series, adapt the exploration template.")
    sample = frame.loc["2005-01":"2024-12", [RETAIL_ID]]
    if sample.empty or sample.index[-1] != pd.Period("2024-12", "M") or pd.isna(sample.iloc[-1,0]):
        raise ValueError("The worked retail exercise needs observations through December 2024. Use the custom template for another sample.")
    sample.index = sample.index.to_timestamp().strftime("%Y-%m-%d")
    return sample.rename_axis("date").reset_index()


def data_structure(prompt, mode="auto", data=None):
    if mode in ("cross_section", "monthly"):
        return mode
    if mode not in ("auto", None):
        raise ValueError("Choose a supported data structure.")
    if re.search(r"cross[- ]section|independent (?:units|households|firms|workers)|one row per (?:person|household|firm|worker)",prompt,re.I):
        return "cross_section"
    if re.search(r"time[- ]series|monthly|reverse[- ]cross|RCF[- ]DML",prompt,re.I):
        return "monthly"
    if re.search(r"\b(?:DML|double machine learning)\b",prompt,re.I):
        if data is None:
            raise ValueError("For DML, specify cross-sectional units or a monthly time series in the prompt or Data structure menu.")
        return "monthly" if "date" in data else "cross_section"
    return "monthly"
