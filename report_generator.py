import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

SOIL_DARK = colors.HexColor("#1B2420")
SAGE = colors.HexColor("#6FA875")
OCHRE = colors.HexColor("#C9973F")
CLASS_COLORS = {
    "Low": colors.HexColor("#C0533E"),
    "Moderate": colors.HexColor("#D9A441"),
    "High": colors.HexColor("#6FA875"),
}
TEXT_MUTED = colors.HexColor("#555555")

FIELD_LABELS = {
    "herd_size": "Herd Size",
    "years_operating": "Years Operating",
    "labor_count": "Number of Workers",
    "capital": "Capital / Investment",
    "feed_inventory_kg": "Feed Inventory (kg)",
    "digital_transaction_freq": "Digital Transactions / Month",
    "market_access_score": "Market Access Score",
    "biosecurity_measures": "Has Biosecurity Measures",
    "monthly_revenue": "Monthly Revenue",
    "monthly_expenses": "Monthly Expenses",
    "loan_amount": "Loan Amount",
}
PESO_FIELDS = {"capital", "monthly_revenue", "monthly_expenses", "loan_amount"}


def _fmt_value(key, value):
    if key in PESO_FIELDS:
        return f"₱ {value:,.2f}"
    if key == "biosecurity_measures":
        return "Yes" if value else "No"
    if key == "market_access_score":
        return f"{value:.0f} / 100"
    return str(value)


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        name="ReportTitle", fontName="Helvetica-Bold", fontSize=18,
        leading=22, textColor=SOIL_DARK, spaceAfter=8, alignment=TA_LEFT,
    ))
    ss.add(ParagraphStyle(
        name="ReportSubtitle", fontName="Helvetica", fontSize=10,
        leading=13, textColor=TEXT_MUTED, spaceBefore=2, spaceAfter=10,
        alignment=TA_LEFT,
    ))
    ss.add(ParagraphStyle(
        name="SectionHead", fontName="Helvetica-Bold", fontSize=12,
        textColor=SOIL_DARK, spaceBefore=14, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        name="BodyFil", fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#222222"),
    ))
    ss.add(ParagraphStyle(
        name="ClassBadge", fontName="Helvetica-Bold", fontSize=16,
        alignment=TA_CENTER, textColor=colors.white,
    ))
    ss.add(ParagraphStyle(
        name="BigStat", fontName="Helvetica-Bold", fontSize=20,
        alignment=TA_CENTER, textColor=SOIL_DARK,
    ))
    ss.add(ParagraphStyle(
        name="SmallLabel", fontName="Helvetica", fontSize=8.5,
        alignment=TA_CENTER, textColor=TEXT_MUTED,
    ))
    ss.add(ParagraphStyle(
        name="FooterNote", fontName="Helvetica-Oblique", fontSize=8,
        textColor=TEXT_MUTED, alignment=TA_CENTER,
    ))
    return ss


def generate_pdf_report(result: dict, data: dict, output_path: str, farm_name: str = "") -> str:
    
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=1.6 * cm, bottomMargin=1.4 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    story = []

    story.append(Paragraph("Swine Farm Profitability Prediction Report", styles["ReportTitle"]))
    subtitle = "Decision-support tool for hog raisers in Bayugan City"
    if farm_name:
        subtitle = f"{farm_name} &mdash; {subtitle}"
    story.append(Paragraph(subtitle, styles["ReportSubtitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y, %I:%M %p')}",
        styles["FooterNote"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#35443C"), spaceBefore=10, spaceAfter=14))

    cls = result["predicted_class"]
    cls_color = CLASS_COLORS.get(cls, SAGE)

    badge_tbl = Table([[Paragraph(f"{cls.upper()} PROFITABILITY", styles["ClassBadge"])]], colWidths=[6.5 * cm])
    badge_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), cls_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    stat_tbl = Table([
        [Paragraph(f"{result['confidence_pct']:.1f}%", styles["BigStat"]),
         Paragraph(f"{result['profit_margin_pct']:.1f}%", styles["BigStat"])],
        [Paragraph("Confidence", styles["SmallLabel"]),
         Paragraph("Profit Margin", styles["SmallLabel"])],
    ], colWidths=[5 * cm, 5 * cm])

    top_row = Table([[badge_tbl, stat_tbl]], colWidths=[7 * cm, 10.4 * cm])
    top_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(top_row)
    story.append(Spacer(1, 4))

    story.append(Paragraph("Probability Breakdown", styles["SectionHead"]))
    proba = result["probability_breakdown"]
    order = ["Low", "Moderate", "High"]
    proba_rows = [["Class", "Probability"]]
    for c in order:
        if c in proba:
            proba_rows.append([c, f"{proba[c]:.1f}%"])
    proba_tbl = Table(proba_rows, colWidths=[8 * cm, 4 * cm])
    proba_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SOIL_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(proba_tbl)

    story.append(Paragraph("Decision-Theory Scores", styles["SectionHead"]))
    score_rows = [
        ["Metric", "Value (0-100)", "Explanation"],
        ["Risk Score", f"{result['risk_score']:.1f}", "Lower is better"],
        ["Expected Utility Score", f"{result['expected_utility_score']:.1f}", "Loss-averse income measure"],
        ["Liquidity Health Index", f"{result['liquidity_health_index']:.1f}", "Higher means healthier cash flow"],
    ]
    score_tbl = Table(score_rows, colWidths=[5 * cm, 3 * cm, 6.5 * cm])
    score_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SOIL_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(score_tbl)

    story.append(Paragraph("Recommendation", styles["SectionHead"]))
    rec = result["decision_recommendation"]
    rec_tbl = Table([[Paragraph(rec, ParagraphStyle(
        name="RecText", fontName="Helvetica-Bold", fontSize=13,
        textColor=CLASS_COLORS.get(cls, SOIL_DARK) if rec == "High Risk" else SOIL_DARK,
    ))]], colWidths=[14.4 * cm])
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F5F2")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(rec_tbl)

    story.append(Paragraph("Explanation", styles["SectionHead"]))
    story.append(Paragraph(result["explanation_fil"], styles["BodyFil"]))

    story.append(Paragraph("Farm Profile Input", styles["SectionHead"]))
    profile_rows = [["Detail", "Value"]]
    for key, label in FIELD_LABELS.items():
        if key in data:
            profile_rows.append([label, _fmt_value(key, data[key])])
    profile_tbl = Table(profile_rows, colWidths=[9 * cm, 5.4 * cm])
    profile_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SOIL_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(profile_tbl)

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceAfter=6))
    story.append(Paragraph(
        "This report is generated by a decision-support machine learning model "
        "(Random Forest) trained using the Resource-Based View Theory and Decision Theory "
        "frameworks. It is a projection only and does not replace professional financial advice.",
        styles["FooterNote"],
    ))

    doc.build(story)
    return output_path


def generate_forecast_pdf_report(
    forecast_rows: list,
    farm_name: str,
    horizon_months: int,
    chart_image_path: str,
    output_path: str,
) -> str:

    styles = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=1.6 * cm, bottomMargin=1.4 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    story = []

    story.append(Paragraph("Income Forecast Report", styles["ReportTitle"]))
    subtitle = f"{horizon_months}-Month Projection — Decision-support tool for hog raisers in Bayugan City"
    if farm_name:
        subtitle = f"{farm_name} &mdash; {subtitle}"
    story.append(Paragraph(subtitle, styles["ReportSubtitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y, %I:%M %p')}",
        styles["FooterNote"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#35443C"), spaceBefore=10, spaceAfter=14))

    story.append(Image(chart_image_path, width=17 * cm, height=6 * cm))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Monthly Projection", styles["SectionHead"]))
    header = ["Month", "Revenue", "Expenses", "Net Income", "Margin"]
    rows = [header]
    for r in forecast_rows:
        rows.append([
            r["month_label"],
            f"₱ {r['projected_revenue']:,.0f}",
            f"₱ {r['projected_expenses']:,.0f}",
            f"₱ {r['projected_net_income']:,.0f}",
            f"{r['projected_profit_margin_pct']:.1f}%",
        ])
    forecast_tbl = Table(rows, colWidths=[3.2 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm, 2.4 * cm])
    forecast_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SOIL_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]))
    story.append(forecast_tbl)

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceAfter=6))
    story.append(Paragraph(
        "This forecast is a projection based on the last prediction, demand seasonality, "
        "and growth patterns of similar farms. It is a projection only and does not "
        "replace professional financial advice.",
        styles["FooterNote"],
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    from predictor_engine import SwineFarmPredictor

    predictor = SwineFarmPredictor()
    sample = {
        "herd_size": 60, "years_operating": 5, "labor_count": 2, "capital": 150000,
        "feed_inventory_kg": 1800, "digital_transaction_freq": 10, "market_access_score": 65,
        "biosecurity_measures": 1, "monthly_revenue": 85000, "monthly_expenses": 60000,
        "loan_amount": 30000,
    }
    result = predictor.predict(sample)
    path = generate_pdf_report(result, sample, "sample_report.pdf", farm_name="Dela Cruz Swine Farm")
    print(f"Report saved -> {path}")
