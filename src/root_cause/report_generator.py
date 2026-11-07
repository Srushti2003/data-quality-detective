import json
import argparse
from pathlib import Path
from datetime import date, datetime

from src.root_cause.rca_engine import get_connection


OUTPUT_DIR = Path("reports/rca")


def json_serializer(value):
    """Convert database date/time values into JSON-compatible strings."""

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return str(value)


def fetch_integrated_findings(conn, batch_id=None):

    query = """
        SELECT
            batch_id,
            feature_name,
            reference_period,
            current_period,
            total_findings,
            high_severity_findings,
            medium_severity_findings,
            low_severity_findings,
            overall_severity,
            investigation_evidence
        FROM integrated_rca_report
    """

    params = None

    if batch_id is not None:
        query += " WHERE batch_id = %s"
        params = (batch_id,)

    query += " ORDER BY batch_id, feature_name"

    with conn.cursor() as cursor:
        cursor.execute(query, params)

        columns = [
            description[0]
            for description in cursor.description
        ]

        rows = cursor.fetchall()

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def calculate_summary(findings):

    summary = {
        "total_feature_groups": len(findings),
        "total_findings": 0,
        "high_severity": 0,
        "medium_severity": 0,
        "low_severity": 0
    }

    for finding in findings:

        summary["total_findings"] += finding["total_findings"]

        summary["high_severity"] += (
            finding["high_severity_findings"]
        )

        summary["medium_severity"] += (
            finding["medium_severity_findings"]
        )

        summary["low_severity"] += (
            finding["low_severity_findings"]
        )

    return summary


def extract_recommendations(findings):

    recommendations = []

    for finding in findings:

        evidence_list = finding["investigation_evidence"] or []

        for evidence in evidence_list:

            recommendation = evidence.get("recommendation")

            if recommendation:

                recommendations.append({
                    "batch_id": finding["batch_id"],
                    "feature_name": finding["feature_name"],
                    "severity": evidence.get("severity"),
                    "recommendation": recommendation
                })

    return recommendations


def build_report(findings, batch_id=None):

    if not findings:
        return None

    periods = sorted({
        (
            finding["reference_period"].isoformat(),
            finding["current_period"].isoformat()
        )
        for finding in findings
    })

    report = {
        "report_title": "Data Quality Detective - RCA Investigation Report",
        "batch_id": batch_id,
        "periods": [
            {
                "reference_period": reference,
                "current_period": current
            }
            for reference, current in periods
        ],
        "generated_at": datetime.now().astimezone().isoformat(),
        "summary": calculate_summary(findings),
        "findings": findings,
        "recommendations": extract_recommendations(findings)
    }

    return report


def save_json_report(report, batch_id):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"rca_report_batch_{batch_id}.json"
        if batch_id is not None
        else "rca_report_all_batches.json"
    )

    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4,
            default=json_serializer
        )

    return output_path


def save_markdown_report(report, batch_id):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"rca_report_batch_{batch_id}.md"
        if batch_id is not None
        else "rca_report_all_batches.md"
    )

    output_path = OUTPUT_DIR / filename

    summary = report["summary"]

    lines = [
        "# Data Quality Detective",
        "## Root Cause Analysis Investigation Report",
        "",
        f"**Batch:** {batch_id if batch_id is not None else 'All batches'}",
        f"**Generated:** {report['generated_at']}",
        "**Investigation periods:**",
        "",
        *[
            f"- Reference: {period['reference_period']} | "
            f"Current: {period['current_period']}"
            for period in report["periods"]
        ],
        "",
        "",
        "## Executive Summary",
        "",
        f"- Feature groups investigated: {summary['total_feature_groups']}",
        f"- Total RCA findings: {summary['total_findings']}",
        f"- High severity: {summary['high_severity']}",
        f"- Medium severity: {summary['medium_severity']}",
        f"- Low severity: {summary['low_severity']}",
        "",
        "## Investigation Findings",
        ""
    ]

    for finding in report["findings"]:

        lines.extend([
            f"### Batch {finding['batch_id']} — {finding['feature_name']}",
            "",
            f"- Overall severity: **{finding['overall_severity']}**",
            f"- Total findings: {finding['total_findings']}",
            ""
        ])

        for evidence in finding["investigation_evidence"] or []:

            lines.extend([
                f"#### {evidence.get('root_cause_category', 'Finding')}",
                "",
                f"- Severity: {evidence.get('severity', 'Not specified')}",
                f"- Confidence: {evidence.get('confidence', 'Not specified')}",
                "",
                "**Description**",
                "",
                evidence.get("description") or "No description available.",
                "",
                "**Recommendation**",
                "",
                evidence.get("recommendation") or "No recommendation available.",
                ""
            ])

    lines.extend([
        "## Recommended Actions",
        ""
    ])

    for index, item in enumerate(report["recommendations"], start=1):

        lines.append(
            f"{index}. **Batch {item['batch_id']} — "
            f"{item['feature_name']} ({item['severity']})**: "
            f"{item['recommendation']}"
        )

    lines.extend([
        "",
        "---",
        "",
        "*This report summarizes recorded investigation evidence. "
        "A detected anomaly or statistical association does not, "
        "by itself, establish the underlying operational cause.*"
    ])

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return output_path


def print_report_summary(report):

    summary = report["summary"]

    print("\n" + "=" * 65)
    print("DATA QUALITY DETECTIVE")
    print("RCA INVESTIGATION REPORT")
    print("=" * 65)

    print(f"\nBatch: {report['batch_id']}")
    print(f"Feature groups: {summary['total_feature_groups']}")
    print(f"Total findings: {summary['total_findings']}")

    print("\nSEVERITY SUMMARY")

    print(f"High   : {summary['high_severity']}")
    print(f"Medium : {summary['medium_severity']}")
    print(f"Low    : {summary['low_severity']}")

    print("\nFEATURE INVESTIGATIONS")

    for finding in report["findings"]:

        print(
            f"\n{finding['feature_name']} | "
            f"Severity: {finding['overall_severity']} | "
            f"Findings: {finding['total_findings']}"
        )

    print("\n" + "=" * 65)


def run_report_generator(batch_id=None):

    conn = get_connection()

    try:

        findings = fetch_integrated_findings(
            conn,
            batch_id
        )

        if not findings:

            print("No RCA findings found for the requested batch.")
            return

        report = build_report(findings, batch_id)

        json_path = save_json_report(report, batch_id)

        markdown_path = save_markdown_report(
            report,
            batch_id
        )

        print_report_summary(report)

        print("\nREPORTS GENERATED SUCCESSFULLY")

        print(f"JSON     : {json_path}")
        print(f"Markdown : {markdown_path}")

    finally:
        conn.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate integrated RCA investigation reports."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch ID to investigate. Omit to report all batches."
    )

    args = parser.parse_args()

    run_report_generator(args.batch)