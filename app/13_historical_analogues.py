"""Display descriptive historical analogues, not risk probabilities."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "youth_unemployment_pct",
    "gdp_growth_pct",
    "inflation_pct",
    "fdi_pct_gdp",
    "trade_pct_gdp",
    "youth_labor_force_participation_pct",
]

LABELS = {
    "youth_unemployment_pct": "Youth unemployment",
    "gdp_growth_pct": "GDP growth",
    "inflation_pct": "Inflation",
    "fdi_pct_gdp": "FDI / GDP",
    "trade_pct_gdp": "Trade / GDP",
    "youth_labor_force_participation_pct": "Youth participation",
}


def render_analogues(panel, country, year, threshold):
    st.subheader("Historical analogues")

    st.caption(
        "Descriptive comparisons using six indicator levels. "
        "Similarity is not a forecast probability or evidence of causation."
    )

    current_rows = panel[
        (panel["country_code"] == country)
        & (panel["year"] == year)
    ]

    if len(current_rows) != 1:
        st.info("A unique reference observation is unavailable.")
        return

    current = current_rows.iloc[0]

    missing = [
        LABELS[name] for name in FEATURES
        if pd.isna(current[name])
    ]

    if missing:
        st.info(
            "Comparison unavailable because these indicators are missing: "
            + ", ".join(missing)
            + ". Choose an earlier reference year."
        )
        return

    u = "youth_unemployment_pct"

    # Historical outcomes must end before the selected reference year.
    pool = panel[
        (panel["year"] + 2 < year)
        & (panel["country_code"] != country)
    ].copy()

    # Join exact calendar years, rather than shifting across missing years.
    for offset in (1, 2):
        future = panel[["country_code", "year", u]].copy()
        future["year"] = future["year"] - offset
        future = future.rename(columns={u: f"u_plus_{offset}"})

        pool = pool.merge(
            future,
            on=["country_code", "year"],
            how="left",
            validate="one_to_one",
        )

    pool = pool.dropna(
        subset=FEATURES + ["u_plus_1", "u_plus_2"]
    ).copy()

    if len(pool) < 20:
        st.info("Too few complete historical cases for this comparison.")
        return

    # Scaling uses only the eligible historical reference pool.
    varying_features = [
        name for name in FEATURES
        if pool[name].nunique() > 1
    ]

    if len(varying_features) < 2:
        st.info("Insufficient variation in the historical reference data.")
        return

    scaler = StandardScaler()
    historical_vectors = scaler.fit_transform(pool[varying_features])
    current_vector = scaler.transform(
        current_rows[varying_features]
    )[0]

    # Root-mean-square standardized distance; lower is closer.
    pool["distance"] = np.sqrt(
        np.mean((historical_vectors - current_vector) ** 2, axis=1)
    )

    pool["max_increase_pp"] = (
        pool[["u_plus_1", "u_plus_2"]].max(axis=1) - pool[u]
    )
    pool["event"] = pool["max_increase_pp"] >= threshold

    # A fixed exploratory cutoff, not a validated similarity threshold.
    matches = (
        pool[pool["distance"] <= 1.0]
        .sort_values(["distance", "country_code", "year"])
        .drop_duplicates("country_code")
        .head(5)
    )

    if matches.empty:
        st.info(
            "No historical cases meet the exploratory distance cutoff. "
            "The app will not force a match."
        )
        return

    display = matches[[
        "country_name",
        "year",
        "distance",
        u,
        "u_plus_1",
        "u_plus_2",
        "max_increase_pp",
        "event",
    ]].rename(columns={
        "country_name": "Country",
        "year": "Historical year",
        "distance": "Distance — lower is closer",
        u: "Unemployment at start (%)",
        "u_plus_1": "One year later (%)",
        "u_plus_2": "Two years later (%)",
        "max_increase_pp": "Maximum increase (pp)",
        "event": f"Increase ≥ {threshold:g} pp",
    })

    st.dataframe(display.round(3), hide_index=True)

    event_count = int(matches["event"].sum())

    st.write(
        f"{event_count} of {len(matches)} selected cases met the "
        f"{threshold:g}-percentage-point event definition."
    )
    st.caption(
        f"Across all {len(pool)} eligible complete historical windows, "
        f"the event frequency was {pool['event'].mean():.1%}. "
        "These overlapping windows are not independent events. "
        "The selected-case frequency is not this country's risk probability."
    )

    with st.expander("Compare the underlying indicators"):
        comparison = [{
            "Case": f"Selected: {current['country_name']} {year}",
            **{LABELS[name]: current[name] for name in FEATURES},
        }]

        for _, row in matches.iterrows():
            comparison.append({
                "Case": f"{row['country_name']} {int(row['year'])}",
                **{LABELS[name]: row[name] for name in FEATURES},
            })

        st.dataframe(pd.DataFrame(comparison).round(3), hide_index=True)

    st.caption(
        "Method: equally weighted standardized indicator levels; "
        "maximum distance 1.0; at most five different countries. "
        "The cutoff is exploratory. Correlated indicators and extreme "
        "values can affect matches. Data are from the current snapshot, "
        "not archived releases available at each historical date."
    )
