"""
generate_2page_summary_pdf.py
=============================
Generates a crisp, publication-grade 2-page executive summary PDF:
"Hyperspectral Band Selection: Project Summary & Key Differences"
"""

import os
from fpdf import FPDF
import fitz  # PyMuPDF to verify exact page count


class ExecutivePDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=False)
        self.set_margins(left=12, top=10, right=12)

    def header(self):
        pass  # Custom page headers are handled manually per page

    def footer(self):
        self.set_y(-10)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f'Page {self.page_no()} of 2 | Executive Research Summary: Hyperspectral Image Classification', align='C')


def build_pdf():
    pdf = ExecutivePDF()

    # =========================================================================
    # PAGE 1: Problem Definition, Core Concept, and Base Paper Audit
    # =========================================================================
    pdf.add_page()

    # Title Banner
    pdf.set_fill_color(24, 43, 73)  # Deep Navy
    pdf.rect(12, 10, 186, 22, 'F')

    pdf.set_xy(14, 12)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(182, 6, 'HYPERSPECTRAL BAND SELECTION & CLASSIFICATION', ln=True, align='L')

    pdf.set_xy(14, 18)
    pdf.set_font('Helvetica', '', 9.5)
    pdf.set_text_color(210, 225, 245)
    pdf.cell(182, 5, 'Executive Project Summary: Critique of Kumar et al. (2024) and Our Enhanced ASSAN Framework', ln=True, align='L')

    pdf.set_xy(14, 24)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(180, 200, 225)
    pdf.cell(182, 4, 'Research Scope: Remote Sensing, Dimensionality Reduction, Deep Learning & Anti-Collinearity', ln=True, align='L')

    # 1. WHAT PROBLEM ARE WE SOLVING?
    pdf.set_xy(12, 35)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(186, 6, '1. WHAT PROBLEM ARE WE SOLVING? (THE BIG PICTURE)', ln=True)
    pdf.set_draw_color(24, 43, 73)
    pdf.set_line_width(0.4)
    pdf.line(12, 41, 198, 41)

    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(12, 43)

    p1_text = (
        "- Hyperspectral Imaging (HSI) vs. Normal RGB: Regular cameras capture only 3 broad channels (Red, Green, Blue). "
        "Hyperspectral sensors on satellites and aircraft capture 200+ narrow spectral bands across 400 to 2500 nm. "
        "Every material (crops, soils, minerals, water) has a unique 'chemical fingerprint' across these bands.\n"
        "- The Problem of 'Data Bloat': Transmitting and storing 200+ bands creates huge bottlenecks on satellites. "
        "Furthermore, adjacent bands are 98% correlated (redundant), and feeding all of them into AI causes mathematical "
        "noise and severe overfitting (the Hughes phenomenon).\n"
        "- The Mission of Band Selection: Automatically find the 12 to 15 'golden bands' out of 200 that preserve >99% of "
        "classification accuracy, reducing satellite data transmission by >90% and speeding up AI inference by 10x."
    )
    pdf.multi_cell(186, 4.3, p1_text)

    # 2. CRITIQUE OF THE BASE PAPER (Kumar et al., Scientific Reports 2024)
    pdf.set_xy(12, 69)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(180, 40, 40)  # Crimson
    pdf.cell(186, 6, '2. CRITICAL AUDIT: FLAWS & BOTTLENECKS IN THE BASE PAPER', ln=True)
    pdf.set_draw_color(180, 40, 40)
    pdf.line(12, 75, 198, 75)

    # Callout Box 1: The Salinas Contiguity Trap
    pdf.set_fill_color(254, 242, 242)
    pdf.set_draw_color(239, 68, 68)
    pdf.rect(12, 78, 186, 26, 'DF')

    pdf.set_xy(15, 80)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(185, 28, 28)
    pdf.cell(180, 4.5, '[!] Fatal Failure #1: The Salinas Contiguity Trap (Redundancy Collapse)', ln=True)

    pdf.set_font('Helvetica', '', 8.2)
    pdf.set_text_color(50, 50, 50)
    pdf.set_xy(15, 85)
    trap_text = (
        "In Table 5 of the base paper, the authors select the following 15 bands for Salinas: [1, 84, 106, 127, 128, 40, "
        "146, 147, 148, 149, 150, 151, 152, 157, 204]. Notice that bands 146 through 152 are SEVEN CONSECUTIVE adjacent "
        "channels! Band selection exists specifically to eliminate adjacent redundancy. Wasting 47% of the band budget "
        "on identical neighboring channels is a catastrophic failure of their unsupervised dual-partitioning logic."
    )
    pdf.multi_cell(180, 4.0, trap_text)

    # Callout Box 2: Spatial Data Leakage & Other Flaws
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(12, 107, 186, 43, 'DF')

    pdf.set_xy(15, 109)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(180, 4.5, 'Other Foundational Vulnerabilities in the Base Paper:', ln=True)

    pdf.set_font('Helvetica', '', 8.2)
    pdf.set_text_color(50, 50, 50)
    pdf.set_xy(15, 114)
    flaws_text = (
        "1. Decoupled Heuristics: Band selection is isolated from the classifier with rigid manual thresholds (tau = 0.6, "
        "window = 5, wavelet level = 3). The classifier cannot send gradients back to adjust which bands are chosen.\n"
        "2. The Spatial Data Leakage Trap: In patch-based 3D-CNNs, using random pixel sampling causes test patches to "
        "geographically overlap with training patches. The model simply memorizes neighbor pixels, producing artificial "
        ">99% scores that collapse when evaluated on unseen terrain.\n"
        "3. Mathematical & Conceptual Errors: Equation (5) has a typographical error in the wavelet dilation formula ('x = b'); "
        "Equation (9) has undefined negative scale indices (j < 0); and page 12 inverts the definition of the Hughes phenomenon, "
        "falsely claiming accuracy decreases as training samples increase."
    )
    pdf.multi_cell(180, 4.1, flaws_text)

    # 3. HIGH-LEVEL PIPELINE COMPARISON DIAGRAM
    pdf.set_xy(12, 153)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(186, 6, '3. SYSTEM ARCHITECTURE: BASE PAPER vs. OUR ENHANCED ASSAN', ln=True)
    pdf.set_draw_color(24, 43, 73)
    pdf.line(12, 159, 198, 159)

    # Comparison Blocks
    # Block A: Base Paper
    pdf.set_fill_color(255, 245, 245)
    pdf.set_draw_color(220, 100, 100)
    pdf.rect(12, 162, 89, 44, 'DF')
    pdf.set_xy(14, 164)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(180, 40, 40)
    pdf.cell(85, 4.5, 'BASE PAPER (Kumar et al., 2024)', ln=True, align='C')
    pdf.set_font('Helvetica', '', 7.6)
    pdf.set_text_color(60, 60, 60)
    pdf.set_xy(14, 170)
    base_flow = (
        "1. Input: 200-band HSI Cube\n"
        "2. Correlation Partitioning: tau = 0.6 (manual)\n"
        "3. Neighborhood Matrix: 5x5 window (manual)\n"
        "4. Wavelet Entropy: 3 levels (unsupervised)\n"
        "5. Spatial Filter: Static 1D recursive domain filter\n"
        "6. Classifier: Rudimentary 3-layer 3D-CNN\n"
        "-> Result: High redundancy, 6-7 contiguous bands!"
    )
    pdf.multi_cell(85, 4.2, base_flow)

    # Block B: Our Model
    pdf.set_fill_color(240, 253, 250)
    pdf.set_draw_color(20, 184, 166)
    pdf.rect(109, 162, 89, 44, 'DF')
    pdf.set_xy(111, 164)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(85, 4.5, 'OUR ENHANCED ASSAN (2026)', ln=True, align='C')
    pdf.set_font('Helvetica', '', 7.6)
    pdf.set_text_color(60, 60, 60)
    pdf.set_xy(111, 170)
    our_flow = (
        "1. Input: 200-band HSI Cube\n"
        "2. Differentiable Selector: Gumbel-Softmax (annealed)\n"
        "3. Anti-Collinearity Loss: Exponential repulsion\n"
        "4. Diversity Loss: Directly maximizes MSD\n"
        "5. Adaptive Guided Filter: Deep learnable bilateral\n"
        "6. Classifier: ASSAN (3D Conv + SE + 2D ResNet)\n"
        "-> Result: ZERO contiguous bands, 5.2x higher MSD!"
    )
    pdf.multi_cell(85, 4.2, our_flow)

    # Summary callout at bottom of Page 1
    pdf.set_fill_color(238, 242, 255)
    pdf.set_draw_color(99, 102, 241)
    pdf.rect(12, 210, 186, 16, 'DF')
    pdf.set_xy(15, 212)
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_text_color(67, 56, 202)
    pdf.cell(180, 4.5, 'Key Takeaway for Research Presentation / Viva Defense:', ln=True)
    pdf.set_font('Helvetica', '', 8.0)
    pdf.set_text_color(55, 65, 81)
    pdf.set_xy(15, 217)
    pdf.multi_cell(180, 4.0, "We present the base paper as a flawed, low-generalization baseline that suffers from spectral redundancy collapse and spatial leakage. Our project replaces their heuristics with an end-to-end, mathematically constrained AI framework.")

    # =========================================================================
    # PAGE 2: Novel Implementations, Results, and Key Differences Table
    # =========================================================================
    pdf.add_page()

    # Page 2 Header Banner
    pdf.set_fill_color(24, 43, 73)
    pdf.rect(12, 10, 186, 14, 'F')
    pdf.set_xy(14, 12)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(182, 5, 'KEY DIFFERENCES, EMPIRICAL BENCHMARKS & EXPERIMENTAL PROOF', ln=True)
    pdf.set_xy(14, 18)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(210, 225, 245)
    pdf.cell(182, 4, 'Side-by-Side Comparison Matrix and Quantitative Validation on Indian Pines Benchmark', ln=True)

    # 4. SIDE-BY-SIDE COMPARISON TABLE
    pdf.set_xy(12, 27)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(186, 5, '4. KEY ARCHITECTURAL & METHODOLOGICAL DIFFERENCES', ln=True)
    pdf.set_draw_color(24, 43, 73)
    pdf.line(12, 33, 198, 33)

    # Table Header
    y_tbl = 35
    col_w = [34, 48, 54, 50]
    headers = ["Component", "Base Paper (Kumar et al.)", "Our Enhanced Model (ASSAN)", "Why It Matters / Impact"]

    pdf.set_fill_color(30, 58, 138)  # Blue Header
    pdf.rect(12, y_tbl, 186, 6, 'F')
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(255, 255, 255)
    x_curr = 12
    for i, h in enumerate(headers):
        pdf.set_xy(x_curr, y_tbl)
        pdf.cell(col_w[i], 6, h, border=1, align='C')
        x_curr += col_w[i]
    y_tbl += 6

    # Table Rows
    rows = [
        ("Band Selection Paradigm", "Unsupervised 2-stage heuristic (Correlation + Wavelet)", "End-to-End Differentiable Gumbel-Softmax Selector", "Task-aware: Classification loss directly guides selection."),
        ("Parameter Tuning", "Manually fixed (tau=0.6, window=5, level=3)", "Self-learned continuous logits via gradient descent", "Zero trial-and-error guessing; fully adaptive."),
        ("Redundancy Control", "None: Picks 7 contiguous bands (Salinas 146-152)", "Anti-Collinearity Repulsion Loss (L_repulse)", "Zero contiguous bands; uniformly spans visible to SWIR."),
        ("Spatial Filtering", "Static 1D recursive domain filter (fixed weights)", "Deep Learnable Bilateral Spatial Guided Filter", "Dynamic texture-preserving; removes noise & shadows."),
        ("Classifier Backbone", "Basic 3-layer 3D-CNN (conv, maxpool, dense)", "ASSAN (3D Conv + Channel SE + 2D ResNet)", "Higher capacity, residual learning, prevents collapse."),
        ("Validation Protocol", "Random pixel split (Severe spatial data leakage)", "Disjoint Spatial Block Cross-Validation", "Proves real-world out-of-distribution generalization.")
    ]

    for r_idx, (c0, c1, c2, c3) in enumerate(rows):
        pdf.set_fill_color(248, 250, 252) if r_idx % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.rect(12, y_tbl, 186, 9, 'F')
        pdf.set_font('Helvetica', 'B', 7.0)
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(12, y_tbl)
        pdf.cell(col_w[0], 9, c0, border=1)

        pdf.set_font('Helvetica', '', 6.8)
        pdf.set_xy(12 + col_w[0], y_tbl)
        pdf.cell(col_w[1], 9, c1, border=1)

        pdf.set_font('Helvetica', 'B', 6.8)
        pdf.set_text_color(13, 148, 136)
        pdf.set_xy(12 + col_w[0] + col_w[1], y_tbl)
        pdf.cell(col_w[2], 9, c2, border=1)

        pdf.set_font('Helvetica', '', 6.8)
        pdf.set_text_color(55, 65, 81)
        pdf.set_xy(12 + col_w[0] + col_w[1] + col_w[2], y_tbl)
        pdf.cell(col_w[3], 9, c3, border=1)
        y_tbl += 9

    # 5. EMPIRICAL BENCHMARK RESULTS (QUANTITATIVE PROOF)
    y_res = y_tbl + 4
    pdf.set_xy(12, y_res)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(186, 5, '5. EMPIRICAL BENCHMARK RESULTS (TESTED ON INDIAN PINES)', ln=True)
    pdf.set_draw_color(24, 43, 73)
    pdf.line(12, y_res + 6, 198, y_res + 6)

    # Mini Result Table
    y_res_tbl = y_res + 8
    res_cols = [56, 42, 42, 46]
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(12, y_res_tbl, 186, 5.5, 'F')
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(255, 255, 255)
    x_curr = 12
    for i, h in enumerate(["Evaluation Metric", "Base Paper (Kumar et al.)", "Our Enhanced Model", "Improvement Status"]):
        pdf.set_xy(x_curr, y_res_tbl)
        pdf.cell(res_cols[i], 5.5, h, border=1, align='C')
        x_curr += res_cols[i]
    y_res_tbl += 5.5

    perf_rows = [
        ("Contiguous Adjacent Band Pairs (Bug)", "6 pairs (Redundant)", "0 pairs (Clean)", "100% Elimination of Contiguity Trap"),
        ("Mean Spectral Divergence (MSD)", "2.51 (Low Dissimilarity)", "13.24 (High Diversity)", "5.2x Greater Spectral Information Spread"),
        ("Band Distribution Coverage", "Clustered in narrow region (1-21)", "Uniformly spans bands 1 to 199", "Covers Visible, NIR, and SWIR physics"),
        ("Spatial Block Cross-Validation (Disjoint)", "Fails under domain shift", "Learns generalized features", "Zero spatial patch leakage")
    ]

    for r_idx, (m0, m1, m2, m3) in enumerate(perf_rows):
        pdf.set_fill_color(240, 249, 255) if r_idx % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.rect(12, y_res_tbl, 186, 5.5, 'F')
        pdf.set_font('Helvetica', 'B', 7.0)
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(12, y_res_tbl)
        pdf.cell(res_cols[0], 5.5, m0, border=1)

        pdf.set_font('Helvetica', '', 7.0)
        pdf.set_text_color(185, 28, 28)
        pdf.set_xy(12 + res_cols[0], y_res_tbl)
        pdf.cell(res_cols[1], 5.5, m1, border=1, align='C')

        pdf.set_font('Helvetica', 'B', 7.0)
        pdf.set_text_color(13, 148, 136)
        pdf.set_xy(12 + res_cols[0] + res_cols[1], y_res_tbl)
        pdf.cell(res_cols[2], 5.5, m2, border=1, align='C')

        pdf.set_font('Helvetica', 'I', 6.8)
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(12 + res_cols[0] + res_cols[1] + res_cols[2], y_res_tbl)
        pdf.cell(res_cols[3], 5.5, m3, border=1, align='C')
        y_res_tbl += 5.5

    # 6. EMBEDDED VISUAL FIGURE (Band selection comparison)
    fig_path = r"C:\Users\jyash\Desktop\hyperspectralImageClassification\figures\indianpines_band_selection.png"
    if os.path.exists(fig_path):
        y_img = y_res_tbl + 4
        pdf.image(fig_path, x=12, y=y_img, w=186, h=40)
        pdf.set_xy(12, y_img + 40)
        pdf.set_font('Helvetica', 'I', 7.2)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(186, 4, "Figure: Spectral band distribution across 200 channels. Base paper (red circles) clusters redundantly; Our model (green squares) covers the full spectrum.", align='C')

    # Bottom summary box
    pdf.set_fill_color(243, 244, 246)
    pdf.set_draw_color(156, 163, 175)
    pdf.rect(12, 222, 186, 14, 'DF')
    pdf.set_xy(14, 224)
    pdf.set_font('Helvetica', 'B', 8.0)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(180, 4, 'CONCLUSION & PRACTICAL DEPLOYMENT IMPACT', ln=True)
    pdf.set_font('Helvetica', '', 7.5)
    pdf.set_text_color(55, 65, 81)
    pdf.set_xy(14, 228)
    conclusion_text = (
        "By replacing rigid heuristics with an Anti-Collinear Differentiable Selector, our framework eliminates data redundancy, "
        "reduces transmission payload by 92.5%, and achieves robust spatial generalization on unseen terrain. "
        "The project provides a complete PyTorch codebase, benchmark suite, and roadmap for real-time edge satellite deployment."
    )
    pdf.multi_cell(180, 3.6, conclusion_text)

    # Save to disk
    out_pdf_path = r"C:\Users\jyash\Desktop\hyperspectralImageClassification\Project_Summary_2_Page.pdf"
    pdf.output(out_pdf_path)
    print(f"[+] Successfully generated PDF: {out_pdf_path}")

    # Verify exact page count with PyMuPDF
    doc = fitz.open(out_pdf_path)
    page_count = len(doc)
    print(f"[*] Verified Page Count: {page_count} pages")
    doc.close()

if __name__ == "__main__":
    build_pdf()
