
import json
import argparse
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)


REPORT_DIR = Path("reports/rca")
PDF_DIR = REPORT_DIR / "pdf"


def load_report(batch_id=None):

    filename = (
        f"rca_report_batch_{batch_id}.json"
        if batch_id is not None
        else "rca_report_all_batches.json"
    )

    path = REPORT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Report not found: {path}. "
            "Generate the JSON report first."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_styles():

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=12,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="FindingHeading",
        parent=styles["Heading3"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#24476B"),
        spaceBefore=8,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name="SmallCell",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10
    ))

    return styles


def make_table(data, widths=None):

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),

        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("FONTSIZE", (0, 0), (-1, -1), 8),

        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),

        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F5F9")]),

        ("LEFTPADDING", (0, 0), (-1, -1), 7),

        ("RIGHTPADDING", (0, 0), (-1, -1), 7),

        ("TOPPADDING", (0, 0), (-1, -1), 7),

        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    return table


def footer(canvas, doc):

    canvas.saveState()

    width, height = A4

    canvas.setStrokeColor(colors.lightgrey)

    canvas.line(
        18 * mm,
        15 * mm,
        width - 18 * mm,
        15 * mm
    )

    canvas.setFont("Helvetica", 8)

    canvas.setFillColor(colors.grey)

    canvas.drawString(
        18 * mm,
        10 * mm,
        "Data Quality Detective | RCA Investigation"
    )

    canvas.drawRightString(
        width - 18 * mm,
        10 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


def build_pdf(report, batch_id=None):

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"rca_report_batch_{batch_id}.pdf"
        if batch_id is not None
        else "rca_report_all_batches.pdf"
    )

    output_path = PDF_DIR / filename

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="Data Quality Detective RCA Report",
        author="Data Quality Detective"
    )

    styles = create_styles()

    story = []

    summary = report["summary"]

    # COVER / HEADER

    story.append(Spacer(1, 25 * mm))

    story.append(Paragraph(
        "DATA QUALITY DETECTIVE",
        styles["ReportTitle"]
    ))

    story.append(Paragraph(
        "ROOT CAUSE ANALYSIS REPORT",
        styles["Heading2"]
    ))

    story.append(Spacer(1, 8 * mm))

    batch_label = (
        f"Batch {batch_id}"
        if batch_id is not None
        else "All batches"
    )

    story.append(Paragraph(
        f"<b>Investigation scope:</b> {batch_label}",
        styles["BodyText"]
    ))

    story.append(Paragraph(
        f"<b>Generated:</b> {report['generated_at']}",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 15 * mm))

    story.append(Paragraph(
        "EXECUTIVE SUMMARY",
        styles["SectionHeading"]
    ))

    summary_data = [
        ["Metric", "Count"],
        ["Feature groups investigated",
         str(summary["total_feature_groups"])],
        ["Total RCA findings",
         str(summary["total_findings"])],
        ["High severity",
         str(summary["high_severity"])],
        ["Medium severity",
         str(summary["medium_severity"])],
        ["Low severity",
         str(summary["low_severity"])]
    ]

    story.append(make_table(
        summary_data,
        [105 * mm, 45 * mm]
    ))

    story.append(Spacer(1, 8 * mm))

    # INVESTIGATION PERIODS

    story.append(Paragraph(
        "INVESTIGATION PERIODS",
        styles["SectionHeading"]
    ))

    period_data = [["Reference period", "Current period"]]

    for period in report.get("periods", []):

        period_data.append([
            str(period["reference_period"]),
            str(period["current_period"])
        ])

    story.append(make_table(
        period_data,
        [75 * mm, 75 * mm]
    ))

    story.append(PageBreak())

    # FINDINGS

    story.append(Paragraph(
        "DETAILED INVESTIGATION FINDINGS",
        styles["SectionHeading"]
    ))

    for finding in report["findings"]:

        finding_title = (
            f"Batch {finding['batch_id']} — "
            f"{finding['feature_name']}"
        )

        elements = [
            Paragraph(
                finding_title,
                styles["FindingHeading"]
            ),

            Paragraph(
                f"<b>Overall severity:</b> "
                f"{finding['overall_severity']} "
                f"| <b>Recorded findings:</b> "
                f"{finding['total_findings']}",
                styles["BodySmall"]
            )
        ]

        for evidence in finding.get(
            "investigation_evidence", []
        ):

            category = evidence.get(
                "root_cause_category", "Finding"
            )

            description = evidence.get(
                "description", "No description available."
            )

            recommendation = evidence.get(
                "recommendation", "No recommendation available."
            )

            severity = evidence.get(
                "severity", "Not specified"
            )

            confidence = evidence.get(
                "confidence", "Not specified"
            )

            elements.extend([
                Paragraph(
                    f"<b>{category}</b>",
                    styles["BodySmall"]
                ),

                Paragraph(
                    f"<b>Severity:</b> {severity} "
                    f"| <b>Confidence:</b> {confidence}",
                    styles["BodySmall"]
                ),

                Paragraph(
                    f"<b>Evidence:</b> {description}",
                    styles["BodySmall"]
                ),

                Paragraph(
                    f"<b>Recommendation:</b> {recommendation}",
                    styles["BodySmall"]
                ),

                Spacer(1, 3 * mm)
            ])

        story.append(KeepTogether(elements))

    # RECOMMENDATIONS

    story.append(PageBreak())

    story.append(Paragraph(
        "RECOMMENDED ACTIONS",
        styles["SectionHeading"]
    ))

    recommendations = report.get("recommendations", [])

    if recommendations:

        recommendation_data = [
            [
                "Batch",
                "Feature",
                "Severity",
                "Recommended action"
            ]
        ]

        for item in recommendations:

            recommendation_data.append([
                str(item["batch_id"]),
                Paragraph(
                    str(item["feature_name"]),
                    styles["SmallCell"]
                ),
                str(item.get("severity", "N/A")),
                Paragraph(
                    str(item["recommendation"]),
                    styles["SmallCell"]
                )
            ])

        story.append(make_table(
            recommendation_data,
            [15 * mm, 40 * mm, 22 * mm, 75 * mm]
        ))

    else:

        story.append(Paragraph(
            "No recommendations available.",
            styles["BodySmall"]
        ))

    # LIMITATIONS

    story.append(Spacer(1, 12 * mm))

    story.append(Paragraph(
        "INVESTIGATION LIMITATIONS",
        styles["SectionHeading"]
    ))

    story.append(Paragraph(
        "This report summarizes recorded root-cause investigation "
        "findings and supporting evidence. Statistical anomalies "
        "and business-rule mismatches do not independently establish "
        "the underlying operational cause. Finding counts represent "
        "persisted RCA records, not unique affected trips. "
        "Recommendations should be validated by the responsible "
        "data or business team before corrective action.",
        styles["BodySmall"]
    ))

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer
    )

    return output_path


def run_pdf_generator(batch_id=None):

    report = load_report(batch_id)

    output_path = build_pdf(report, batch_id)

    print("\nPDF REPORT GENERATED SUCCESSFULLY")

    print(f"File: {output_path.resolve()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate PDF RCA investigation reports."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None
    )

    args = parser.parse_args()

    run_pdf_generator(args.batch)
