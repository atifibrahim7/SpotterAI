"""Freight rate model: clean -> features -> time-based validation -> final fit -> predictions.

Run: python train.py
Outputs: validation_predictions.csv, december-chart-inputs.csv (filled), metrics.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TRAIN = "train-test.csv"
VALIDATION = "validation.csv"
TEMPLATE = "validation-predictions-template.csv"
DECEMBER = "december-chart-inputs.csv"

# City names are left out on purpose: 12% of validation rows involve a city never seen in
# training, so lanes are described by coordinates, which generalise to new cities.
FEATURES = [
    "equipment", "distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "market_index", "mi_missing", "quote_signal", "month", "dow", "doy",
]
# Rate-per-mile / equipment median sits in 0.7-1.5 for 98.5% of rows; the rest form separate
# clusters at ~0.1-0.5x and ~2-6x (decimal/entry errors). Keep only the normal band.
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 2.0
SEED = 42


def clean_features(df: pd.DataFrame, equipment_types: list[str]) -> pd.DataFrame:
    """Same feature cleaning for train and test. Never drops rows."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = df["weight"].abs()  # negative weights are sign errors; NaN stays NaN
    df["mi_missing"] = df["market_index"].isna().astype(int)
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.day_of_week
    df["doy"] = df["date"].dt.day_of_year
    df["equipment"] = pd.Categorical(df["equipment"], categories=equipment_types)
    return df


def drop_label_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rpm = df["posted_rate"] / df["distance"]
    median = rpm.groupby(df["equipment"], observed=True).transform("median")
    ratio = rpm / median
    bad = (ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH)
    return df[~bad], int(bad.sum())


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features="from_dtype", random_state=SEED,
    )


def fit(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    # Target = log(rate per mile): errors become relative, trees learn price/mile not distance scale.
    target = np.log(train["posted_rate"] / train["distance"])
    return new_model().fit(train[FEATURES], target)


def predict(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(df[FEATURES])) * df["distance"].to_numpy()


def metrics(actual: pd.Series, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual.to_numpy()
    return {
        "MAE": round(float(np.abs(err).mean()), 2),
        "MAPE_%": round(float((np.abs(err) / actual.to_numpy()).mean() * 100), 2),
        "RMSE": round(float(np.sqrt((err**2).mean())), 2),
    }


def baseline(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Median rate-per-mile by group x distance; falls back to equipment median for unseen groups."""
    rpm = (train["posted_rate"] / train["distance"]).rename("rpm")
    by_group = rpm.groupby([train[k] for k in keys], observed=True).median()
    by_equipment = rpm.groupby(train["equipment"], observed=True).median()
    looked_up = test.join(by_group, on=keys)["rpm"]
    fallback = test["equipment"].map(by_equipment).astype(float)
    return looked_up.fillna(fallback).to_numpy() * test["distance"].to_numpy()


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, float]]:
    actual = test["posted_rate"]
    return {
        "model": metrics(actual, predict(fit(train), test)),
        "baseline_equipment": metrics(actual, baseline(train, test, ["equipment"])),
        "baseline_lane": metrics(actual, baseline(train, test, ["pickup", "delivery", "equipment"])),
        "rows": {"train": len(train), "test": len(test)},
    }


def main() -> None:
    raw_train = pd.read_csv(TRAIN)
    raw_validation = pd.read_csv(VALIDATION)
    december = pd.read_csv(DECEMBER)

    equipment_types = sorted(raw_train["equipment"].unique())

    report: dict = {
        "data_quality": {
            "train_rows": len(raw_train),
            "negative_weights_fixed": int((raw_train["weight"] < 0).sum()),
            "missing_weight": int(raw_train["weight"].isna().sum()),
            "missing_market_index": int(raw_train["market_index"].isna().sum()),
        }
    }

    train = clean_features(raw_train, equipment_types)
    train, n_outliers = drop_label_outliers(train)
    report["data_quality"]["label_outliers_dropped"] = n_outliers

    # 1) Holdout: train Jan-Aug, test Sep-Oct (mirrors "predict the next 2 months").
    holdout_train = train[train["month"] <= 8]
    holdout_test = train[train["month"] >= 9]
    report["holdout_jan_aug_vs_sep_oct"] = evaluate(holdout_train, holdout_test)

    # 2) Expanding-window backtest: train months 1..k, test months k+1..k+2.
    report["backtest"] = {
        f"train_1-{k}_test_{k + 1}-{k + 2}": evaluate(
            train[train["month"] <= k], train[train["month"].between(k + 1, k + 2)]
        )["model"]
        for k in (4, 6, 8)
    }

    # 3) Final model on all cleaned Jan-Oct data.
    model = fit(train)

    validation = clean_features(raw_validation, equipment_types)
    validation["predicted_rate"] = predict(model, validation).round(2)
    template = pd.read_csv(TEMPLATE)[["load_id"]]
    out = template.merge(validation[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and len(out) == len(template)
    out.to_csv("validation_predictions.csv", index=False)

    # December chart: inputs lack market_index/quote_signal and coordinates.
    # Fill market signals with that date's mean from validation.csv (it covers December),
    # and coordinates from the city lookup.
    daily = raw_validation.groupby("date")[["market_index", "quote_signal"]].mean()
    coords = pd.concat([
        raw_train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        raw_train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates("city").set_index("city")
    dec = december.drop(columns="predicted_rate").join(daily, on="date")
    dec["market_index"] = dec["market_index"].fillna(daily["market_index"].mean())
    dec["quote_signal"] = dec["quote_signal"].fillna(daily["quote_signal"].mean())
    for end in ("pickup", "delivery"):
        dec[f"{end}_lat"] = dec[end].map(coords["lat"])
        dec[f"{end}_lon"] = dec[end].map(coords["lon"])
    dec = clean_features(dec, equipment_types)
    december["predicted_rate"] = predict(model, dec).round(2)
    december.to_csv(DECEMBER, index=False)

    report["december_rate_range"] = [float(december["predicted_rate"].min()),
                                     float(december["predicted_rate"].max())]
    with open("metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
