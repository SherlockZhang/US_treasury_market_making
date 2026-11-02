from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle


OUTPUT = "US_Treasury_Futures_and_Options_Exchanges.pdf"


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#12324A"), alignment=TA_LEFT, spaceAfter=10))
    styles.add(ParagraphStyle(name="SubtitleCustom", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#52606D"), spaceAfter=16))
    styles.add(ParagraphStyle(name="HeadingCustom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#12324A"), spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9, leading=13, textColor=colors.HexColor("#263746"), spaceAfter=7))
    styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#52606D"), spaceAfter=4))
    styles.add(ParagraphStyle(name="TableCustom", parent=styles["BodyText"], fontSize=7.4, leading=9.2, textColor=colors.HexColor("#263746")))
    styles.add(ParagraphStyle(name="TableHeaderCustom", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9.2, textColor=colors.white))
    return styles


def p(text, style):
    return Paragraph(text, style)


def table(data, widths):
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12324A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C4CE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F7")]),
    ]))
    return result


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DDE3"))
    canvas.line(0.65 * inch, 0.54 * inch, 7.85 * inch, 0.54 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6B7785"))
    canvas.drawString(0.65 * inch, 0.36 * inch, "US Treasury futures and options | Prepared 2026-09-28")
    canvas.drawRightString(7.85 * inch, 0.36 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf():
    styles = make_styles()
    doc = SimpleDocTemplate(OUTPUT, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.62 * inch, bottomMargin=0.72 * inch, title="US Treasury Futures and Options: Exchanges and Contract Details", author="OpenAI")
    story = []
    story.append(p("US Treasury Futures and Options", styles["TitleCustom"]))
    story.append(p("Directly listed exchange venue and core contract reference | Information checked 2026-09-28", styles["SubtitleCustom"]))
    story.append(p("Executive summary", styles["HeadingCustom"]))
    story.append(p("The directly verified exchange venue for standardized US Treasury futures and options is the Chicago Board of Trade (CBOT), a designated contract market within CME Group. The products trade electronically on CME Globex and clear through CME Clearing. CME also offers bilateral block, EFR and EFP execution channels, but those are execution methods around the listed contracts rather than separate exchanges.", styles["BodyCustom"]))
    story.append(p("A check of the ICE product page located an ICE Brent crude contract, not a directly listed US Treasury futures or options product. This report therefore does not count ICE, Eurex, Nasdaq, or other venues that may provide access, data, clearing, or related rates products unless a direct US Treasury contract listing was verified.", styles["BodyCustom"]))

    story.append(p("Exchange scope", styles["HeadingCustom"]))
    scope_data = [
        [p("Venue", styles["TableHeaderCustom"]), p("Role", styles["TableHeaderCustom"]), p("Verified scope", styles["TableHeaderCustom"])],
        [p("CBOT / CME Group", styles["TableCustom"]), p("Listed futures and options exchange", styles["TableCustom"]), p("US Treasury note and bond futures; options on those futures; CME Globex, blocks, EFRs and EFPs", styles["TableCustom"])],
        [p("CME Clearing", styles["TableCustom"]), p("Central counterparty / clearing service", styles["TableCustom"]), p("Clears the listed CME Group interest-rate products; it is not a separate listing venue", styles["TableCustom"])],
        [p("ICE / Eurex / Nasdaq", styles["TableCustom"]), p("Not included as direct venues", styles["TableCustom"]), p("No directly listed US Treasury futures/options contract was verified for this report", styles["TableCustom"])],
    ]
    story.append(table(scope_data, [1.25 * inch, 1.55 * inch, 4.4 * inch]))

    story.append(p("Core Treasury futures", styles["HeadingCustom"]))
    story.append(p("All six contracts below are deliverable futures. The deliverable security is selected from a published basket of eligible US Treasury securities; the cheapest-to-deliver security and conversion factor can change by contract month. Prices are quoted in points and fractions of 1/32 of par. Tick values below are the standard outright minimum price fluctuation values.", styles["BodyCustom"]))
    futures_data = [
        [p(x, styles["TableHeaderCustom"]) for x in ["Contract", "CME symbol", "Contract unit", "Minimum tick", "Tick value", "Underlying / delivery"]],
        [p(x, styles["TableCustom"]) for x in ["2-Year Treasury Note", "ZT", "$200,000 face value", "1/4 of 1/32", "$15.625", "Eligible 2-year Treasury note basket; physical delivery"]],
        [p(x, styles["TableCustom"]) for x in ["5-Year Treasury Note", "ZF", "$100,000 face value", "1/4 of 1/32", "$7.8125", "Eligible 5-year Treasury note basket; physical delivery"]],
        [p(x, styles["TableCustom"]) for x in ["10-Year Treasury Note", "ZN / TY", "$100,000 face value", "1/2 of 1/32", "$15.625", "Eligible 10-year Treasury note basket; physical delivery"]],
        [p(x, styles["TableCustom"]) for x in ["Ultra 10-Year Treasury Note", "TN", "$100,000 face value", "1/2 of 1/32", "$15.625", "Longer-duration eligible note basket; physical delivery"]],
        [p(x, styles["TableCustom"]) for x in ["US Treasury Bond", "ZB", "$100,000 face value", "1/32", "$31.25", "Eligible long-term Treasury bond basket; physical delivery"]],
        [p(x, styles["TableCustom"]) for x in ["Ultra US Treasury Bond", "UB", "$100,000 face value", "1/32", "$31.25", "Longer-duration bond basket; physical delivery"]],
    ]
    story.append(table(futures_data, [1.3 * inch, 0.7 * inch, 1.1 * inch, 0.9 * inch, 0.65 * inch, 2.55 * inch]))

    story.append(PageBreak())
    story.append(p("Options on Treasury futures", styles["HeadingCustom"]))
    story.append(p("CBOT lists options on each of the core Treasury futures families. One option contract gives exposure to one underlying Treasury futures contract. These are options on futures, not options on a cash Treasury bond: exercise or assignment results in a futures position, subject to the applicable contract rules.", styles["BodyCustom"]))
    options_data = [
        [p(x, styles["TableHeaderCustom"]) for x in ["Option family", "Underlying future", "Contract multiplier", "Exercise / expiry convention", "Premium and settlement"]],
        [p(x, styles["TableCustom"]) for x in ["2-Year Treasury Note options", "ZT", "1 ZT future", "American-style options on futures; listed serial and quarterly expiries vary by series", "Quoted in Treasury price points; premium paid in dollars; exercise creates futures exposure"]],
        [p(x, styles["TableCustom"]) for x in ["5-Year Treasury Note options", "ZF", "1 ZF future", "American-style options on futures; listed serial and quarterly expiries vary by series", "Quoted in Treasury price points; premium paid in dollars; exercise creates futures exposure"]],
        [p(x, styles["TableCustom"]) for x in ["10-Year Treasury Note options", "ZN / TY", "1 ZN future", "American-style options on futures; listed serial and quarterly expiries vary by series", "Quoted in Treasury price points; premium paid in dollars; exercise creates futures exposure"]],
        [p(x, styles["TableCustom"]) for x in ["Ultra 10-Year Treasury Note options", "TN", "1 TN future", "American-style options on futures; listed serial and quarterly expiries vary by series", "Quoted in Treasury price points; premium paid in dollars; exercise creates futures exposure"]],
        [p(x, styles["TableCustom"]) for x in ["US Treasury Bond options", "ZB", "1 ZB future", "American-style options on futures; listed serial and quarterly expiries vary by series", "Quoted in Treasury price points; premium paid in dollars; exercise creates futures exposure"]],
        [p(x, styles["TableCustom"]) for x in ["Ultra US Treasury Bond options", "UB", "1 UB future", "American-style options on futures; listed serial and quarterly expiries vary by series", "Quoted in Treasury price points; premium paid in dollars; exercise creates futures exposure"]],
    ]
    story.append(table(options_data, [1.2 * inch, 0.8 * inch, 0.95 * inch, 2.25 * inch, 2.0 * inch]))

    story.append(p("Trading and contract mechanics", styles["HeadingCustom"]))
    mechanics_data = [
        [p("Item", styles["TableHeaderCustom"]), p("Practical detail", styles["TableHeaderCustom"])],
        [p("Trading venue", styles["TableCustom"]), p("CME Globex central limit order book; block trades and RFQ liquidity are also available under CME rules.", styles["TableCustom"])],
        [p("Trading hours", styles["TableCustom"]), p("Near 24-hour electronic trading with a daily maintenance window. Consult the current CME holiday and trading-hours calendar before scheduling orders.", styles["TableCustom"])],
        [p("Futures settlement", styles["TableCustom"]), p("Daily mark-to-market. Deliverable futures converge through physical delivery of an eligible Treasury security, with invoice price determined by futures price, conversion factor and accrued interest.", styles["TableCustom"])],
        [p("Options settlement", styles["TableCustom"]), p("Options are premium-style options on futures. Exercise and assignment result in the underlying futures contract; final dates and exercise cutoffs are contract-specific.", styles["TableCustom"])],
        [p("Contract months", styles["TableCustom"]), p("Quarterly Treasury futures cycles are standard. Options also include serial and other listed expiries; the live contract calendar controls availability.", styles["TableCustom"])],
        [p("Risk note", styles["TableCustom"]), p("Tick values, eligible baskets, delivery months, price limits, expiries and hours can change. Confirm the live CME contract specification and rulebook before trading or valuing a position.", styles["TableCustom"])],
    ]
    story.append(table(mechanics_data, [1.25 * inch, 5.95 * inch]))

    story.append(p("Sources and methodology", styles["HeadingCustom"]))
    sources = [
        "CME Group, U.S. Treasury futures, options and cash: https://www.cmegroup.com/markets/interest-rates/us-treasury.html",
        "CME Group, Treasury futures and options contract specifications: https://www.cmegroup.com/trading/interest-rates/us-treasury-futures-contract-specifications.html",
        "CME Group, CBOT rulebook: https://www.cmegroup.com/rulebook/CBOT/",
        "CME Group, daily settlements and contract calendars: https://www.cmegroup.com/market-data/daily-settlements.html",
        "ICE product page checked for scope comparison: https://www.ice.com/products/219/US-Treasury-Futures (resolved to Brent crude in the current catalog)",
    ]
    for source in sources:
        story.append(p(source, styles["SmallCustom"]))
    story.append(p("Methodology: product names, symbols, contract units and tick values are summarized from the CME Treasury product family and standard contract specifications. The report is a reference guide, not a trading recommendation. Product specifications and listed expiries are subject to exchange change notices.", styles["SmallCustom"]))
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build_pdf()