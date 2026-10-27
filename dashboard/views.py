"""Turn API payloads into display-ready tables and charts (no Streamlit calls, so it is testable)."""
import altair as alt
import pandas as pd


def predictions_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["drivers"]).rename(columns={
        "name": "Driver", "constructor_name": "Team", "predicted_points": "Predicted",
        "interval_low": "Low", "interval_high": "High", "price": "Price ($M)",
        "points_per_million": "Pts per $M"})
    return df[["Driver", "Team", "Predicted", "Low", "High", "Price ($M)", "Pts per $M", "price_source"]]


def has_intervals(payload: dict) -> bool:
    return payload.get("interval_level") is not None


def interval_chart(df: pd.DataFrame, level: float) -> alt.Chart:
    """Predicted points as a dot, with the calibrated interval as a bar behind it."""
    order = df["Driver"].tolist()  # already sorted by predicted points
    base = alt.Chart(df).encode(y=alt.Y("Driver:N", sort=order, title=None))
    bars = base.mark_rule(strokeWidth=6, opacity=0.35).encode(
        x=alt.X("Low:Q", title="Fantasy points"), x2="High:Q",
        tooltip=["Driver", "Team", "Low", "Predicted", "High"])
    dots = base.mark_point(filled=True, size=70, color="#e10600").encode(x="Predicted:Q", tooltip=["Driver", "Predicted"])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[3, 3], color="grey").encode(x="x:Q")
    return (zero + bars + dots).properties(
        height=24 * len(df), title=f"Predicted points with {level:.0%} interval")


def team_frame(payload: dict) -> pd.DataFrame:
    rows = [{"Type": "Driver", "Name": d["name"], "Price ($M)": d["price"],
             "Predicted": d["predicted_points"], "DRS": "x2" if d["drs"] else ""} for d in payload["drivers"]]
    rows += [{"Type": "Constructor", "Name": c["name"], "Price ($M)": c["price"],
              "Predicted": c["predicted_points"], "DRS": ""} for c in payload["constructors"]]
    return pd.DataFrame(rows)


def metrics_frame(payload: dict) -> pd.DataFrame:
    def ci(m, key, digits):
        lo, hi = m[f"{key}_ci"]
        return f"{m[key]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

    return pd.DataFrame([{"Model": m["model"], "MAE": ci(m, "mae", 2), "MSE": ci(m, "mse", 1),
                          "Spearman": ci(m, "spearman", 3), "Top-5 capture": ci(m, "top5_capture", 3)}
                         for m in payload["models"]])
