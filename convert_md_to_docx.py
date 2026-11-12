"""
convert_md_to_docx.py
=====================
Converts the Beginner's Guide & Viva Walkthrough markdown into a professionally
formatted Microsoft Word document (.docx).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)


def add_callout_box(doc, title, text, bg_color="F1F5F9", border_color="1E3A8A"):
    """Adds a stylish callout box with a colored left border and soft background."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Set thick left border and none for others
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(title + "\n")
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(11)
    run_t.font.color.rgb = RGBColor.from_string(border_color)

    run_b = p.add_run(text)
    run_b.font.name = "Calibri"
    run_b.font.size = Pt(10)
    run_b.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def create_document():
    doc = docx.Document()

    # Set Standard 1-inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Document Header / Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    r_title = p_title.add_run("Beginner's Guide & Viva Walkthrough")
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(22)
    r_title.font.color.rgb = RGBColor(30, 58, 138)  # Deep Navy

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("Companion Guide to 2-Page Executive Summary PDF & Defense Cheat-Sheet")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(15, 118, 110)  # Teal
    r_sub.italic = True

    # Target metadata
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(14)
    r_meta = p_meta.add_run("Target PDF: ")
    r_meta.bold = True
    r_meta.font.size = Pt(9.5)
    r_meta2 = p_meta.add_run(r"C:\Users\jyash\Desktop\hyperspectralImageClassification\Project_Summary_2_Page.pdf")
    r_meta2.font.size = Pt(9.5)
    r_meta2.font.color.rgb = RGBColor(100, 116, 139)

    # Section 1: Why This Document Was Created
    h1 = doc.add_heading("Why This 2-Page Document Was Created", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    h1.runs[0].font.color.rgb = RGBColor(30, 58, 138)

    p_intro = doc.add_paragraph(
        "If you are explaining your project to someone who has never heard of remote sensing or AI, "
        "this 2-page document makes everything crystal-clear in under 2 minutes:"
    )
    p_intro.paragraph_format.space_after = Pt(6)

    doc.add_paragraph("1. Real-World Analogies: Uses simple concepts like phone cameras vs. satellite chemical eyes, and buying 7 identical math textbooks.", style='List Bullet')
    doc.add_paragraph("2. Clean Professional Layout: Zero overlapping text, color-coded cards, and comfortable readability.", style='List Bullet')
    doc.add_paragraph("3. Clear Academic Audit: Demonstrates exactly why the 2024 Nature Scientific Reports paper made a critical mistake and how our engineering fix solved it.", style='List Bullet')

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Section 2: PAGE 1 Breakdown
    h2 = doc.add_heading("PAGE 1: The Basics, The Problem & The Base Paper's Mistake", level=1)
    h2.paragraph_format.space_before = Pt(16)
    h2.paragraph_format.space_after = Pt(6)
    h2.runs[0].font.color.rgb = RGBColor(30, 58, 138)

    # 1. Title Banner
    h3_1 = doc.add_heading("1. Title Banner (Top Navy Card)", level=2)
    h3_1.runs[0].font.color.rgb = RGBColor(15, 118, 110)
    p = doc.add_paragraph()
    r = p.add_run("What it says: ")
    r.bold = True
    p.add_run('"Hyperspectral Band Selection: Beginner Guide — How We Solved Data Bloat & Fixed a Critical Mistake in a 2024 Nature Scientific Reports Paper"\n')
    r = p.add_run("How to explain it: ")
    r.bold = True
    p.add_run('"Our project is about teaching AI to make satellite imaging faster, lighter, and smarter by cutting out 92% of unnecessary data without losing accuracy."')

    # 2. Box 1: What is Hyperspectral Imaging?
    h3_2 = doc.add_heading("2. Box 1: What is Hyperspectral Imaging? (In Simple Words)", level=2)
    h3_2.runs[0].font.color.rgb = RGBColor(15, 118, 110)
    doc.add_paragraph("• Phone Camera (3 Colors): Captures only Red, Green, and Blue light. In a normal photo, healthy crops and diseased crops both look like ordinary green leaves.")
    doc.add_paragraph("• Hyperspectral Camera (200+ Colors): Satellites and agricultural drones carry special sensors that split light into over 200 narrow invisible color bands across 400 to 2500 nanometers.")
    doc.add_paragraph("• The Superpower ('Chemical Fingerprints'): Every material on Earth (corn, soybean, dry soil, water, plastic, minerals) reflects light in a unique pattern across these 200 bands. This acts like a chemical barcode identifying materials invisible to human eyes.")

    # 3. Box 2: The Problem
    h3_3 = doc.add_heading("3. Box 2: The Problem: Too Much Heavy & Repetitive Data", level=2)
    h3_3.runs[0].font.color.rgb = RGBColor(15, 118, 110)
    doc.add_paragraph("• Problem A (File Size Bloat): 200 color bands per picture means each image is gigabytes in size. Satellites orbiting in space have limited radio bandwidth and cannot send this mountain of data down to Earth fast enough.")
    doc.add_paragraph("• Problem B (Duplicate Bands): Band 50 (e.g. 710 nm) and Band 51 (715 nm) are 99% identical! Feeding 200 repetitive bands into an AI model creates mathematical confusion, slows training down by 10x, and leads to errors (called the Hughes phenomenon).")
    doc.add_paragraph("• The Band Selection Mission: Can we automatically find the 15 'Golden Bands' out of 200 that hold 99% of the useful information? If yes, we shrink file sizes by 92% and make AI run 10x faster!")

    # 4. Box 3: The Salinas Trap
    h3_4 = doc.add_heading("4. Box 3: The Base Paper's Fatal Mistake: The Salinas Duplicate Trap", level=2)
    h3_4.runs[0].font.color.rgb = RGBColor(185, 28, 28)

    callout_salinas = (
        "In 2024, researchers published a paper in Nature Scientific Reports (Kumar et al.) claiming to solve band selection.\n\n"
        "When we inspected Table 5 in their paper for the Salinas dataset, their algorithm picked these 15 bands:\n"
        "    [1, 84, 106, 127, 128, 40,  146, 147, 148, 149, 150, 151, 152,  157, 204]\n\n"
        "Look at bands 146 through 152: That is SEVEN bands right next to each other in a row!\n\n"
        "The Textbook Analogy: Band selection exists specifically to eliminate duplicates! Picking 7 neighbor bands is like "
        "being given money to buy 15 school textbooks and buying 7 copies of the exact same math book. You wasted almost half "
        "your budget on identical pages!"
    )
    add_callout_box(doc, "FATAL AUDIT FINDING: THE SALINAS CONTIGUITY TRAP", callout_salinas, bg_color="FEF2F2", border_color="EF4444")

    # 5. Box 4: Summary of Discoveries
    h3_5 = doc.add_heading("5. Box 4: Summary of What We Discovered About the Base Paper", level=2)
    h3_5.runs[0].font.color.rgb = RGBColor(15, 118, 110)
    doc.add_paragraph("1. It gets trapped in duplicate neighbor bands (Salinas had 7 adjacent bands; Indian Pines had 6).", style='List Bullet')
    doc.add_paragraph("2. It uses manual guesswork (a fixed threshold of 0.60) instead of letting AI learn what is useful.", style='List Bullet')
    doc.add_paragraph("3. It cheated in testing: By randomly picking pixels, test patches overlapped with train patches (spatial data leakage), artificially inflating their accuracy to 99%.", style='List Bullet')

    # Page Break for Page 2
    doc.add_page_break()

    # Section 3: PAGE 2 Breakdown
    h2_p2 = doc.add_heading("PAGE 2: Our Smart Fixes, The Comparison Table & Verified Results", level=1)
    h2_p2.paragraph_format.space_before = Pt(14)
    h2_p2.paragraph_format.space_after = Pt(6)
    h2_p2.runs[0].font.color.rgb = RGBColor(15, 118, 110)

    # Box 1: Our 3 Core Engineering Fixes
    h3_fix = doc.add_heading("1. Box 1: Our 3 Simple Engineering Fixes (The Res-SSAN Framework)", level=2)
    h3_fix.runs[0].font.color.rgb = RGBColor(30, 58, 138)

    doc.add_paragraph("1. The Anti-Duplicate Penalty (Anti-Collinearity Rule): We added a mathematical penalty rule to the AI training loss. If the AI tries to pick two bands that are close together and similar, it gets penalized! This strictly forces the AI to spread its picks across the whole spectrum (Visible, NIR, and SWIR).")
    doc.add_paragraph("2. Connected AI (End-to-End Differentiable): Instead of using a manual 0.60 guess, the classifier AI is directly connected to the band selector. The classifier tells the selector: 'Pick bands that help me tell corn from soybean!'")
    doc.add_paragraph("3. Fair Real-World Testing (Spatial Blocks): We test on completely separate farm plots with zero overlap. This proves our AI actually learns real plant biology rather than memorizing neighbor pixels.")

    # Table: Comparison Table
    h3_tbl = doc.add_heading("2. Simple Comparison Table: Base Paper vs. Our Model", level=2)
    h3_tbl.runs[0].font.color.rgb = RGBColor(30, 58, 138)

    tbl = doc.add_table(rows=6, cols=4)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    col_widths = [Inches(1.3), Inches(1.8), Inches(1.9), Inches(1.5)]
    headers = ["Feature", "Base Paper (Kumar et al.)", "Our Enhanced Model", "Why It Matters"]

    # Format Header Row
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "1E3A8A")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    table_data = [
        ("Band Selection", "Manual guess (tau = 0.60)", "AI-guided (learns automatically)", "No guesswork; adapts to any camera."),
        ("Duplicate Bands", "Picks 7 adjacent bands in Salinas!", "ZERO adjacent bands picked!", "No wasted satellite bandwidth or RAM."),
        ("Band Spread", "Clumped into a tiny corner (1-21)", "Evenly spreads across 1 to 199", "Covers visible light, NIR, and SWIR."),
        ("AI Network", "Old 3-layer basic 3D-CNN", "Res-SSAN (3D Conv + Attention)", "Higher accuracy, handles complex crops."),
        ("Fair Testing", "Random pixels (leaks answers)", "Separate spatial ground blocks", "Proves it works on new, unseen land.")
    ]

    for row_idx, data in enumerate(table_data):
        row_cells = tbl.rows[row_idx + 1].cells
        bg = "F8FAFC" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(data):
            row_cells[col_idx].width = col_widths[col_idx]
            set_cell_background(row_cells[col_idx], bg)
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[col_idx].paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            if col_idx == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(30, 41, 59)
            elif col_idx == 1:
                r.font.color.rgb = RGBColor(185, 28, 28)
            elif col_idx == 2:
                r.bold = True
                r.font.color.rgb = RGBColor(15, 118, 110)
            else:
                r.font.color.rgb = RGBColor(71, 85, 105)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Box 3: The Scorecard Numbers
    h3_score = doc.add_heading("3. The Scorecard: Verified Numbers from Our Project Run", level=2)
    h3_score.runs[0].font.color.rgb = RGBColor(30, 58, 138)

    doc.add_paragraph("• Duplicate Adjacent Bands (The Bug): Base paper had 6 duplicates (Indian Pines) and 7 (Salinas). Our Model achieved 0 (Zero duplicate bands!).")
    doc.add_paragraph("• Information Diversity Score (MSD): Base paper scored 2.51 (Low diversity). Our Model scored 13.24 (over 5.2x higher diversity!).")
    doc.add_paragraph("• Bands Selected Across 200 Channels: Base paper clumped [1, 2, 4, 8, 9, 11, 12, 14, 15, 17, 18, 19, 21, 95, 139]. Our Model evenly spaced [1, 16, 26, 42, 56, 69, 87, 100, 115, 128, 144, 156, 171, 182, 199].")
    doc.add_paragraph("• Satellite Radio Payload Saved: Keeping 15 bands instead of 200 saves 92.5% of satellite transmission data!")

    # Box 4: Viva Cheat-Sheet
    h3_viva = doc.add_heading("4. Viva Cheat-Sheet: What to Say in Your 60-Second Defense", level=2)
    h3_viva.runs[0].font.color.rgb = RGBColor(67, 56, 202)

    viva_speech = (
        '"Sir/Madam, satellites take pictures in over 200 narrow color bands called hyperspectral images. '
        'These files are gigabytes in size, and neighboring bands are 99% duplicate data.\n\n'
        'Our project solves Band Selection: finding the 15 most informative bands to cut file size by 92% '
        'without losing classification accuracy.\n\n'
        'We audited a 2024 paper from Nature Scientific Reports and found a major flaw: on the Salinas dataset, '
        'it accidentally picked 7 duplicate neighbor bands in a row (146 to 152), wasting half its budget.\n\n'
        'To fix this, we created an AI Anti-Duplicate penalty rule that mathematically forces the AI to choose diverse '
        'bands spread across the entire spectrum, and connected it directly to an Attention Neural Network.\n\n'
        'Our verified project run achieved ZERO duplicate bands, 5.2x higher information diversity, and proved it can '
        'identify crops on completely unseen farm fields."'
    )
    add_callout_box(doc, "60-SECOND VIVA SPEECH (MEMORIZE OR READ THIS)", viva_speech, bg_color="EEF2FF", border_color="6366F1")

    # Output path
    out_dir = r"C:\Users\jyash\Downloads\KUNAL HYPERSPECTRAL PROJECT"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Beginners_Guide_and_Viva_Walkthrough.docx")
    doc.save(out_path)
    print(f"[+] Successfully created Word document: {out_path}")

    # Also copy the PDF file into the same directory so Kunal has both together
    src_pdf = r"C:\Users\jyash\Desktop\hyperspectralImageClassification\Project_Summary_2_Page.pdf"
    dst_pdf = os.path.join(out_dir, "Project_Summary_2_Page.pdf")
    if os.path.exists(src_pdf):
        import shutil
        shutil.copyfile(src_pdf, dst_pdf)
        print(f"[+] Also copied 2-Page PDF into: {dst_pdf}")


if __name__ == "__main__":
    create_document()
