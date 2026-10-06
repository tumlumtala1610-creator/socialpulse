"""SocialPulse research dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="SocialPulse",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 SocialPulse")
st.caption("Youth employment data explorer and forecasting research")

config_path = ROOT / "docs" / "experiment_config.json"

if not config_path.exists():
    st.error("Missing experiment configuration.")
    st.stop()

with config_path.open(encoding="utf-8") as file:
    config = json.load(file)

run_id = config["source_run"]
panel_path = (
    ROOT / "data" / "processed" / run_id / "country_year_panel.csv"
)
report_dir = ROOT / "reports" / run_id

if not panel_path.exists():
    st.error("Prepared data is missing. Run 04_build_panel.py first.")
    st.stop()

panel = pd.read_csv(panel_path)

if panel.empty or panel.duplicated(["country_code", "year"]).any():
    st.error("The panel is empty or contains duplicate country-years.")
    st.stop()

if not panel["source_run"].eq(run_id).all():
    st.error("The data does not match the configured source run.")
    st.stop()

labels = {
    "youth_unemployment_pct": "Youth unemployment (% of youth labor force)",
    "gdp_growth_pct": "GDP growth (annual %)",
    "inflation_pct": "Consumer price inflation (annual %)",
    "fdi_pct_gdp": "FDI net inflows (% of GDP)",
    "trade_pct_gdp": "Trade (% of GDP)",
    "youth_labor_force_participation_pct": (
        "Youth labor force participation (% of population ages 15–24)"
    ),
}

countries = (
    panel[["country_code", "country_name"]]
    .drop_duplicates()
    .sort_values("country_name")
)
country_names = dict(
    zip(countries["country_code"], countries["country_name"])
)
country_codes = countries["country_code"].tolist()

selected_country = st.sidebar.selectbox(
    "Country / economy",
    country_codes,
    index=country_codes.index("NLD") if "NLD" in country_codes else 0,
    format_func=lambda code: country_names[code],
)

selected_year = st.sidebar.selectbox(
    "Reference year",
    sorted(panel["year"].unique(), reverse=True),
)

st.sidebar.caption(
    "Reference year describes the data, not its publication date."
)

country_data = panel[
    panel["country_code"] == selected_country
].sort_values("year")

selected = country_data[
    country_data["year"] == selected_year
].iloc[0]

st.subheader(f"{country_names[selected_country]} · {selected_year}")

st.info(
    "Research prototype. The model has not demonstrated reliable "
    "early-warning performance on the reserved test years. "
    "This dashboard does not assign operational risk categories."
)

overview, quality, evaluation = st.tabs([
    "Country overview",
    "Data quality",
    "Model evaluation",
])

with overview:
    columns = st.columns(3)

    for column, indicator in zip(
        columns,
        ["youth_unemployment_pct", "gdp_growth_pct", "inflation_pct"],
    ):
        value = selected[indicator]
        column.metric(
            labels[indicator],
            "No data" if pd.isna(value) else f"{value:.2f}%",
        )

    indicator = st.selectbox(
        "Historical indicator",
        list(labels),
        format_func=lambda name: labels[name],
    )

    history = country_data[
        country_data["year"] <= selected_year
    ][["year", indicator]].copy()

    st.caption(labels[indicator])

    if history[indicator].notna().any():
        st.line_chart(history.set_index("year"), height=340)
    else:
        st.write("No available values for this indicator and period.")

    st.caption(
        "Youth unemployment uses ILO modeled estimates. "
        "A low unemployment rate does not capture all forms of "
        "youth employment difficulty."
    )

    display_data = country_data[
        ["year", *labels.keys()]
    ].rename(columns=labels)

    st.dataframe(display_data, hide_index=True)

    st.download_button(
        "Download this country's data",
        data=country_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_country}_historical_data.csv",
        mime="text/csv",
    )

with quality:
    st.write(f"Availability in reference year {selected_year}")

    quality_rows = []
    for indicator, label in labels.items():
        available = pd.notna(selected[indicator])
        quality_rows.append({
            "Indicator": label,
            "Status": "Available" if available else "Missing",
        })

    st.dataframe(pd.DataFrame(quality_rows), hide_index=True)

    st.caption(
        "Available means the source contains a value. "
        "It does not imply a directly observed value or availability "
        "at a past forecast date. Missing values are not zero."
    )

    manifest_path = ROOT / "data" / "raw" / run_id / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        st.write("Download completed at UTC:", manifest["finished_at_utc"])

    st.caption(f"Dataset snapshot: {run_id}")

with evaluation:
    threshold = config["primary_threshold_pp"]

    st.write(
        f"Target: youth unemployment rises by at least {threshold:g} "
        "percentage points in either of the following two years."
    )
    st.write(
        "Selected model: logistic regression using the current level, "
        "one-year change and three-year trend of youth unemployment."
    )

    metrics_path = report_dir / "final_test" / "test_metrics.csv"
    top_path = report_dir / "final_test" / "top10_by_year.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)

        st.subheader("Reserved test results")
        st.dataframe(metrics, hide_index=True)

        st.caption(
            "Higher average precision is better for ranking. "
            "Lower Brier score and log loss are better for probability "
            "prediction. Compare models within the same year."
        )

        if top_path.exists():
            top = pd.read_csv(top_path)
            summary = top.groupby("origin_year").agg(
                selected_windows=("target_primary", "size"),
                positive_windows=("target_primary", "sum"),
            )
            summary["precision_at_10"] = (
                summary["positive_windows"] / summary["selected_windows"]
            )
            st.write("Results among the ten highest-ranked countries")
            st.dataframe(summary)

        st.warning(
            "In the first final evaluation, neither year's top 10 "
            "contained a positive event. Average predicted probabilities "
            "also exceeded observed event rates. This does not support "
            "presenting the model as a validated warning system."
        )
    else:
        st.write("Final test reports are not available.")

    st.caption(
        "Evaluation uses the current historical data snapshot, "
        "not archived releases available at each past forecast date. "
        "The two test years contain few positive windows, and overlapping "
        "windows are not independent events."
    )

st.caption("Source: World Bank Indicators API · Research prototype")
