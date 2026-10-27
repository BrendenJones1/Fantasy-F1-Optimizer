"""Streamlit dashboard over the f1opt API: next-race predictions, the optimal team, model quality.

Run from the repo root with the API up:  python -m streamlit run dashboard/app.py
The API address comes from API_URL (default http://localhost:8010).
"""
import streamlit as st

from dashboard.api_client import ApiClient, ApiError
from dashboard.views import has_intervals, interval_chart, metrics_frame, predictions_frame, team_frame

st.set_page_config(page_title="Fantasy F1 Optimizer", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def load_predictions(base_url: str, level: float) -> dict:
    return ApiClient(base_url).predictions(level)


@st.cache_data(ttl=60, show_spinner=False)
def load_team(base_url: str, budget: float) -> dict:
    return ApiClient(base_url).team(budget)


@st.cache_data(ttl=60, show_spinner=False)
def load_metrics(base_url: str) -> dict:
    return ApiClient(base_url).metrics()


client = ApiClient()
with st.sidebar:
    st.header("Settings")
    level = st.slider("Prediction interval", 0.5, 0.95, 0.8, 0.05, format="%.2f",
                      help="Share of outcomes the interval should contain.")
    if st.button("Refresh data"):
        st.cache_data.clear()
    st.caption(f"API: {client.base_url}")

st.title("Fantasy F1 Optimizer")
try:
    predictions = load_predictions(client.base_url, level)
except ApiError as err:
    st.error(err.detail)
    if err.status in (404, 503):
        st.caption("The service is up but has no race or model yet. The scheduler needs to finish its first cycle.")
    st.stop()

race = predictions["race"]
st.subheader(f"{race['name']} {race['season']} (round {race['round']}, {race['date']}"
             f"{', sprint weekend' if race['is_sprint_weekend'] else ''})")
st.caption(f"Model: {predictions['model']}, trained through season {predictions['trained_through']['season']} "
           f"round {predictions['trained_through']['round']}.")

df = predictions_frame(predictions)
if "demo" in set(df["price_source"].dropna()):
    st.warning("Prices are demo values derived from recent form, not the game's official prices. "
               "Load real prices with scripts/load_prices.py.")

tab_pred, tab_team, tab_model = st.tabs(["Predictions", "Team", "Model quality"])

with tab_pred:
    if has_intervals(predictions):
        st.altair_chart(interval_chart(df, predictions["interval_level"]), width="stretch")
        st.caption("Intervals are calibrated on past races: about "
                   f"{predictions['interval_level']:.0%} of actual driver scores fall inside. They are wide because "
                   "retirements and safety cars cannot be predicted.")
    else:
        st.info("No prediction intervals yet: the model has too little history to calibrate them.")
    st.dataframe(df.drop(columns="price_source"), hide_index=True, width="stretch", column_config={
        "Predicted": st.column_config.NumberColumn(format="%.1f"),
        "Low": st.column_config.NumberColumn(format="%.1f"),
        "High": st.column_config.NumberColumn(format="%.1f"),
        "Price ($M)": st.column_config.NumberColumn(format="%.1f"),
        "Pts per $M": st.column_config.NumberColumn(format="%.2f")})

with tab_team:
    budget = st.number_input("Budget cap ($M)", min_value=50.0, max_value=500.0, value=100.0, step=0.5)
    try:
        team = load_team(client.base_url, budget)
    except ApiError as err:
        (st.info if err.status == 409 else st.warning if err.status == 422 else st.error)(err.detail)
    else:
        cols = st.columns(3)
        greedy = team["greedy_baseline_points"]
        cols[0].metric("Expected points", f"{team['expected_points']:.1f}",
                       None if greedy is None else f"{team['expected_points'] - greedy:+.1f} vs greedy")
        cols[1].metric(f"Total price (cap ${team['budget']:g}M)", f"${team['total_price']:.1f}M")
        cols[2].metric("DRS driver", next(d["name"] for d in team["drivers"] if d["drs"]))
        st.dataframe(team_frame(team), hide_index=True, width="stretch")
        if greedy is None:
            st.caption("The greedy baseline (best points per $M first) could not fill a legal roster within this budget.")
        st.caption("Picked by an exact budget-constrained optimisation over predicted points. The expected total "
                   "is a forecast: the uncertainty of a whole team is not modelled.")

with tab_model:
    try:
        metrics = load_metrics(client.base_url)
    except ApiError as err:
        st.info(err.detail)
    else:
        st.write(metrics["backtest_description"])
        st.dataframe(metrics_frame(metrics), hide_index=True, width="stretch")
        meta = metrics.get("deployed_model_metadata") or {}
        st.caption(f"Deployed model: {metrics['deployed_model']}. {metrics['backtest_races']} backtest races; "
                   f"MAE and MSE are in Fantasy points.")
        if meta.get("calibration"):
            st.caption(f"Intervals: {meta['calibration']['method']}, from {meta['calibration']['n_residuals']} "
                       "out-of-sample residuals.")
