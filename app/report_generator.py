from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


def generate_pdf_report(
    analysis,
    results,
    related_results,
    source="Requirement Analysis",
    explanation=""
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#12325c"),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#12325c"),
        spaceBefore=10,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15
    )

    story = []

    story.append(
        Paragraph(
            "BISense AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Procurement Standards Analysis Report",
            styles["Heading3"]
        )
    )

    story.append(Spacer(1, 12))

    # -----------------------------
    # Analysis Summary
    # -----------------------------

    story.append(
        Paragraph(
            "Procurement Analysis",
            heading_style
        )
    )

    specs = analysis.get("specifications", [])

    if isinstance(specs, list):
        specs = ", ".join(specs)

    compliance = analysis.get(
        "compliance_needs",
        []
    )

    if isinstance(compliance, list):
        compliance = ", ".join(compliance)

    analysis_data = [
        ["Source", source],
        [
            "Product",
            analysis.get(
                "product",
                "Not identified"
            )
        ],
        [
            "Category",
            analysis.get(
                "category",
                "Not identified"
            )
        ],
        [
            "Application",
            analysis.get(
                "application",
                "Not identified"
            )
        ],
        [
            "Specifications",
            specs or "Not identified"
        ],
        [
            "Compliance Needs",
            compliance or "Not identified"
        ]
    ]

    analysis_table = Table(
        analysis_data,
        colWidths=[
            45 * mm,
            115 * mm
        ]
    )

    analysis_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#eaf2ff")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d7e5f8")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(analysis_table)

    # -----------------------------
    # Recommended Standards
    # -----------------------------

    story.append(
        Paragraph(
            "Recommended Standards",
            heading_style
        )
    )

    if results:

        standards_data = [
            [
                "Standard",
                "Title",
                "Match",
                "Certification"
            ]
        ]

        for result in results:

            standards_data.append([
                str(
                    result.get(
                        "standard_id",
                        ""
                    )
                ),
                str(
                    result.get(
                        "title",
                        ""
                    )
                ),
                f"{result.get('score', 0)}%",
                str(
                    result.get(
                        "certification",
                        ""
                    )
                    or "N/A"
                )
            ])

        standards_table = Table(
            standards_data,
            colWidths=[
                35 * mm,
                75 * mm,
                25 * mm,
                30 * mm
            ],
            repeatRows=1
        )

        standards_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#12325c")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#dbe4ef")
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )

        story.append(standards_table)

    else:

        story.append(
            Paragraph(
                "No recommended standards found.",
                body_style
            )
        )

    # -----------------------------
    # Allied Standards
    # -----------------------------

    story.append(
        Paragraph(
            "Allied Standards",
            heading_style
        )
    )

    if related_results:

        for standard in related_results:

            story.append(
                Paragraph(
                    f"<b>{standard.get('standard_id', '')}</b> "
                    f"- {standard.get('title', '')}",
                    body_style
                )
            )

            story.append(
                Paragraph(
                    f"Category: "
                    f"{standard.get('category', '')}",
                    body_style
                )
            )

            story.append(Spacer(1, 5))

    else:

        story.append(
            Paragraph(
                "No additional allied standards identified.",
                body_style
            )
        )

    # -----------------------------
    # AI Explanation
    # -----------------------------

    if explanation:

        story.append(
            Paragraph(
                "AI Recommendation Explanation",
                heading_style
            )
        )

        safe_explanation = (
            explanation
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        story.append(
            Paragraph(
                safe_explanation,
                body_style
            )
        )

    # -----------------------------
    # Disclaimer
    # -----------------------------

    story.append(
        Paragraph(
            "Verification Notice",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "This report is generated by the BISense AI prototype. "
            "Recommendations are based on the prototype standards knowledge base "
            "and must be verified against authoritative BIS sources before use "
            "in real procurement or compliance decisions.",
            body_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer