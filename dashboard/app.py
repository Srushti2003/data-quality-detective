
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

import json
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Data Quality Detective | Drift Analytics",
    page_icon="🔍",
    layout="wide",
)

# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

@st.cache_resource
def get_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME"),
    )

    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=60)
def load_drift_results():
    query = text("""
        SELECT
            current_batch_id,
            current_period,
            feature_name,
            metric_name,
            metric_value,
            threshold,
            drift_detected,
            severity
        FROM drift_results
        WHERE current_batch_id IN (3, 4)
        ORDER BY current_batch_id, feature_name, metric_name
    """)

    with get_engine().connect() as connection:
        return pd.read_sql(query, connection)


@st.cache_data(ttl=60)
def load_distribution_data(batch_id):
    query = text("""
        SELECT
            passenger_count,
            trip_distance,
            fare_amount,
            total_amount,
            tip_amount,
            tolls_amount,
            congestion_surcharge,
            airport_fee,
            cbd_congestion_fee
        FROM taxi_trips
        WHERE batch_id = :batch_id
    """)

    with get_engine().connect() as connection:
        return pd.read_sql(
            query,
            connection,
            params={"batch_id": batch_id},
        )


# --------------------------------------------------
# RCA REPORT LOADING
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RCA_REPORT_DIR = PROJECT_ROOT / "reports" / "rca"

RCA_PDF_DIR = RCA_REPORT_DIR / "pdf"


@st.cache_data(ttl=60)
def load_rca_report(batch_id=None):

    filename = (
        f"rca_report_batch_{batch_id}.json"
        if batch_id is not None
        else "rca_report_all_batches.json"
    )

    report_path = RCA_REPORT_DIR / filename

    if not report_path.exists():
        return None

    with open(report_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_report_file(batch_id, extension):

    if batch_id is None:
        filename = f"rca_report_all_batches.{extension}"
    else:
        filename = f"rca_report_batch_{batch_id}.{extension}"

    if extension == "pdf":
        return RCA_PDF_DIR / filename

    return RCA_REPORT_DIR / filename

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    drift_df = load_drift_results()

except Exception as error:
    st.error(f"Could not connect to PostgreSQL: {error}")
    st.stop()

if drift_df.empty:
    st.warning("No drift results found for February and March.")
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🔍 Data Quality Detective")

st.sidebar.caption("AI-powered data reliability platform")

selected_month = st.sidebar.selectbox(
    "Select comparison month",
    options=[3, 4],
    format_func=lambda x: {
        3: "February 2025",
        4: "March 2025",
    }[x],
)

month_df = drift_df[
    drift_df["current_batch_id"] == selected_month
].copy()

st.sidebar.divider()

st.sidebar.markdown("**Reference dataset**")
st.sidebar.write("January 2025")

st.sidebar.markdown("**Current dataset**")
st.sidebar.write(
    "February 2025" if selected_month == 3
    else "March 2025"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Data Drift Analytics")
st.caption(
    "Monitor statistical distribution changes "
    "across monthly NYC Yellow Taxi datasets."
)

st.divider()

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

total_checks = len(month_df)

drifted_checks = int(
    month_df["drift_detected"].sum()
)

affected_features = int(
    month_df.loc[
        month_df["drift_detected"],
        "feature_name"
    ].nunique()
)

total_features = int(
    month_df["feature_name"].nunique()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Drift Checks",
    total_checks,
)

col2.metric(
    "Drift Flags",
    drifted_checks,
)

col3.metric(
    "Affected Features",
    affected_features,
)

col4.metric(
    "Features Monitored",
    total_features,
)

st.divider()

# --------------------------------------------------
# DRIFT SUMMARY
# --------------------------------------------------

st.subheader("Drift Detection Summary")

summary = (
    month_df.groupby("metric_name")
    .agg(
        total_features=("feature_name", "count"),
        drifted_features=("drift_detected", "sum"),
    )
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        summary,
        x="metric_name",
        y="drifted_features",
        color="metric_name",
        title="Drifted Features by Metric",
        text="drifted_features",
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Detection Metric",
        yaxis_title="Drifted Features",
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(
        summary,
        names="metric_name",
        values="drifted_features",
        title="Distribution of Drift Flags",
        hole=0.55,
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FEATURE-METRIC HEATMAP
# --------------------------------------------------

st.subheader("Feature-Level Drift Heatmap")

heatmap_data = month_df.pivot_table(
    index="feature_name",
    columns="metric_name",
    values="drift_detected",
    aggfunc="max",
).astype(float)

fig = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    color_continuous_scale=[
        [0, "#E5E7EB"],
        [1, "#DC2626"],
    ],
    labels={
        "x": "Detection Metric",
        "y": "Feature",
        "color": "Drift Flag",
    },
    title="Detected Drift by Feature and Metric",
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# DETAILED RESULTS
# --------------------------------------------------

st.subheader("Detailed Drift Results")

display_df = month_df.copy()

display_df["metric_value"] = (
    display_df["metric_value"].round(6)
)

display_df["threshold"] = (
    display_df["threshold"].round(4)
)

st.dataframe(
    display_df[
        [
            "feature_name",
            "metric_name",
            "metric_value",
            "threshold",
            "drift_detected",
            "severity",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# FEATURE DISTRIBUTION COMPARISON
# --------------------------------------------------

st.divider()

st.subheader("Distribution Comparison")

feature_options = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "tolls_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

selected_feature = st.selectbox(
    "Select feature to investigate",
    feature_options,
)

with st.spinner("Loading monthly distributions..."):
    jan_df = load_distribution_data(1)
    current_df = load_distribution_data(selected_month)

jan_values = pd.to_numeric(
    jan_df[selected_feature],
    errors="coerce",
).dropna()

current_values = pd.to_numeric(
    current_df[selected_feature],
    errors="coerce",
).dropna()

plot_df = pd.DataFrame({
    "value": pd.concat(
        [jan_values, current_values],
        ignore_index=True,
    ),
    "month": (
        ["January 2025"] * len(jan_values)
        + [
            "February 2025" if selected_month == 3
            else "March 2025"
        ] * len(current_values)
    ),
})

fig = px.histogram(
    plot_df,
    x="value",
    color="month",
    barmode="overlay",
    opacity=0.65,
    marginal="box",
    title=f"{selected_feature}: January vs Current Month",
    histnorm="probability density",
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# INVESTIGATION RECOMMENDATIONS
# --------------------------------------------------

st.divider()

st.subheader("Investigation Recommendations")

flagged_features = (
    month_df[month_df["drift_detected"]]
    ["feature_name"]
    .unique()
)

if len(flagged_features) == 0:
    st.success(
        "No features crossed the configured drift thresholds."
    )

else:
    for feature in flagged_features:
        feature_metrics = month_df[
            (month_df["feature_name"] == feature)
            & (month_df["drift_detected"])
        ]["metric_name"].tolist()

        st.markdown(
            f"**{feature}** — Drift detected by: "
            f"{', '.join(feature_metrics)}"
        )

        if feature == "trip_distance":
            st.write(
                "Investigate changes in trip lengths, "
                "pickup/dropoff patterns and geographic demand."
            )

        elif feature in ["fare_amount", "total_amount"]:
            st.write(
                "Investigate fare structure, trip distances, "
                "surcharges and pricing changes."
            )

        elif feature == "tip_amount":
            st.write(
                "Investigate changes in payment behaviour "
                "and the relationship between tips and fares."
            )

        elif feature == "cbd_congestion_fee":
            st.write(
                "Investigate changes in the distribution of "
                "CBD congestion charges and applicable trips."
            )

        else:
            st.write(
                "Compare the feature distribution against "
                "the January reference and investigate "
                "potential business or data-pipeline changes."
            )



# --------------------------------------------------
# ROOT CAUSE ANALYSIS DASHBOARD
# --------------------------------------------------

st.divider()

st.title("Root Cause Analysis")
st.caption(
    "Consolidated investigation findings, supporting evidence "
    "and recommended corrective actions."
)

rca_report = load_rca_report(selected_month)

if rca_report is None:

    st.warning(
        "RCA report not found. Generate the report before "
        "viewing investigation results."
    )

else:

    rca_summary = rca_report["summary"]

    st.subheader("RCA Investigation Overview")

    st.caption(
        "Counts represent persisted RCA investigation records, "
        "not unique affected trips."
    )

    rca_col1, rca_col2, rca_col3, rca_col4 = st.columns(4)

    rca_col1.metric(
        "Feature Groups",
        rca_summary["total_feature_groups"]
    )

    rca_col2.metric(
        "Total RCA Findings",
        rca_summary["total_findings"]
    )

    rca_col3.metric(
        "High Severity",
        rca_summary["high_severity"]
    )

    rca_col4.metric(
        "Medium Severity",
        rca_summary["medium_severity"]
    )

    st.divider()

    # ----------------------------------------------
    # SEVERITY DISTRIBUTION
    # ----------------------------------------------

    st.subheader("RCA Severity Distribution")

    severity_df = pd.DataFrame({
        "Severity": ["High", "Medium", "Low"],
        "Findings": [
            rca_summary["high_severity"],
            rca_summary["medium_severity"],
            rca_summary["low_severity"]
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            severity_df,
            x="Severity",
            y="Findings",
            color="Severity",
            text="Findings",
            title="Findings by Severity"
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Severity",
            yaxis_title="Number of RCA Records"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            severity_df,
            names="Severity",
            values="Findings",
            hole=0.55,
            title="RCA Severity Composition"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ----------------------------------------------
    # FEATURE-LEVEL INVESTIGATION
    # ----------------------------------------------

    st.subheader("Feature-Level RCA Investigations")

    findings = rca_report.get("findings", [])

    finding_df = pd.DataFrame([
        {
            "Feature": item["feature_name"],
            "Severity": item["overall_severity"].title(),
            "RCA Records": item["total_findings"],
            "High": item["high_severity_findings"],
            "Medium": item["medium_severity_findings"],
            "Low": item["low_severity_findings"]
        }
        for item in findings
    ])

    st.dataframe(
        finding_df,
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------------
    # DETAILED RCA EVIDENCE
    # ----------------------------------------------

    st.subheader("Investigation Evidence")

    feature_options = sorted({
        item["feature_name"]
        for item in findings
    })

    selected_rca_feature = st.selectbox(
        "Select feature to investigate",
        feature_options,
        key="rca_feature_selector"
    )

    selected_findings = [
        item for item in findings
        if item["feature_name"] == selected_rca_feature
    ]

    for finding in selected_findings:

        evidence_list = finding.get(
            "investigation_evidence", []
        ) or []

        for index, evidence in enumerate(evidence_list):

            category = evidence.get(
                "root_cause_category",
                "Investigation finding"
            )

            severity = evidence.get(
                "severity",
                "Not specified"
            )

            confidence = evidence.get(
                "confidence",
                "Not specified"
            )

            with st.expander(
                f"{category} | Severity: {severity.title()} "
                f"| Confidence: {confidence.title()}",
                expanded=False
            ):

                st.markdown("**Investigation description**")

                st.write(
                    evidence.get(
                        "description",
                        "No description available."
                    )
                )

                st.markdown("**Recommended action**")

                st.write(
                    evidence.get(
                        "recommendation",
                        "No recommendation available."
                    )
                )

                st.markdown("**Supporting evidence**")

                st.json(
                    evidence.get(
                        "supporting_evidence",
                        {}
                    )
                )

                st.caption(
                    f"Drift metric: "
                    f"{evidence.get('drift_metric', 'N/A')}"
                )

    st.divider()

    # ----------------------------------------------
    # RECOMMENDED ACTIONS
    # ----------------------------------------------

    st.subheader("Recommended Corrective Actions")

    recommendations = rca_report.get(
        "recommendations",
        []
    )

    if recommendations:

        for index, item in enumerate(
            recommendations,
            start=1
        ):

            st.markdown(
                f"**{index}. {item['feature_name']}** "
                f"— Batch {item['batch_id']} "
                f"— {item.get('severity', 'N/A').title()}"
            )

            st.write(item["recommendation"])

    else:

        st.info("No recommendations are available.")

    st.divider()

    # ----------------------------------------------
    # REPORT DOWNLOADS
    # ----------------------------------------------

    st.subheader("Download RCA Investigation Reports")

    st.caption(
        "Download the generated reports for the selected batch."
    )

    download_col1, download_col2, download_col3 = st.columns(3)

    report_formats = [
        ("json", "JSON"),
        ("md", "Markdown"),
        ("pdf", "PDF")
    ]

    columns = [
        download_col1,
        download_col2,
        download_col3
    ]

    for column, (extension, label) in zip(
        columns,
        report_formats
    ):

        report_path = get_report_file(
            selected_month,
            extension
        )

        with column:

            if report_path.exists():

                mime_types = {
                    "json": "application/json",
                    "md": "text/markdown",
                    "pdf": "application/pdf"
                }

                st.download_button(
                    label=f"Download {label}",
                    data=report_path.read_bytes(),
                    file_name=report_path.name,
                    mime=mime_types[extension],
                    key=f"download_rca_{extension}_{selected_month}",
                    use_container_width=True
                )

            else:

                st.warning(
                    f"{label} report not found."
                )

    # ----------------------------------------------
    # ALL-BATCH REPORT DOWNLOADS
    # ----------------------------------------------

    with st.expander("All-batches RCA report"):

        all_report = load_rca_report()

        if all_report:

            st.write(
                f"Total RCA records: "
                f"{all_report['summary']['total_findings']}"
            )

            all_col1, all_col2, all_col3 = st.columns(3)

            all_columns = [
                all_col1,
                all_col2,
                all_col3
            ]

            for column, (extension, label) in zip(
                all_columns,
                report_formats
            ):

                report_path = get_report_file(
                    None,
                    extension
                )

                with column:

                    if report_path.exists():

                        mime_types = {
                            "json": "application/json",
                            "md": "text/markdown",
                            "pdf": "application/pdf"
                        }

                        st.download_button(
                            label=f"Download {label}",
                            data=report_path.read_bytes(),
                            file_name=report_path.name,
                            mime=mime_types[extension],
                            key=f"download_all_rca_{extension}",
                            use_container_width=True
                        )

        else:

            st.warning(
                "All-batches RCA report not found."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Data Quality Detective | Drift Analytics | "
    "Reference: January 2025 | "
    "Metrics: PSI, KS Test, Wasserstein, Jensen-Shannon"
)
