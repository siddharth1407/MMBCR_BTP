"""Generate a self-contained, beautifully styled printable HTML report
conforming to IIT Guwahati B.Tech Project Report Guidelines.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "BTP_REPORT.md"
HTML_OUT = ROOT / "docs" / "BTP_Report_Printable.html"

def build_printable_html():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BTP Report: AI-Based Kinetic Modeling of MMBCR - Siddharth Patel</title>
<!-- KaTeX for math rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
        ]
    }});"></script>
<style>
@page {{
    size: A4;
    margin-top: 30mm;
    margin-bottom: 22mm;
    margin-left: 30mm;
    margin-right: 20mm;
    @bottom-center {{
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 11pt;
    }}
}}

body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 12pt;
    line-height: 1.5;
    color: #111;
    background: #f8f9fa;
    margin: 0;
    padding: 20px;
}}

.page-container {{
    max-width: 800px;
    margin: 0 auto;
    background: #fff;
    padding: 50px 60px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}

@media print {{
    body {{
        background: #fff;
        padding: 0;
    }}
    .page-container {{
        max-width: 100%;
        margin: 0;
        padding: 0;
        box-shadow: none;
    }}
    .no-print {{
        display: none !important;
    }}
    .page-break {{
        page-break-before: always;
        break-before: page;
    }}
}}

h1.report-title {{
    font-size: 20pt;
    text-align: center;
    line-height: 1.3;
    margin-top: 40px;
    margin-bottom: 30px;
    font-weight: bold;
    text-transform: uppercase;
}}

h1.chapter-title {{
    font-size: 18pt;
    text-align: center;
    margin-top: 50px;
    margin-bottom: 25px;
    font-weight: bold;
    page-break-before: always;
    break-before: page;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
}}

h2 {{
    font-size: 15pt;
    margin-top: 30px;
    margin-bottom: 12px;
    font-weight: bold;
    color: #1a1a1a;
}}

h3 {{
    font-size: 13pt;
    margin-top: 20px;
    margin-bottom: 8px;
    font-weight: bold;
    color: #2c3e50;
}}

p {{
    text-align: justify;
    margin-top: 0;
    margin-bottom: 14px;
    text-indent: 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 11pt;
}}

table, th, td {{
    border: 1px solid #ddd;
}}

th {{
    background-color: #f2f4f8;
    color: #222;
    font-weight: bold;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid #aaa;
}}

td {{
    padding: 8px 12px;
    vertical-align: top;
}}

tr:nth-child(even) {{
    background-color: #fafbfc;
}}

img {{
    max-width: 95%;
    height: auto;
    display: block;
    margin: 20px auto 8px auto;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
}}

.figure-caption {{
    text-align: center;
    font-size: 10.5pt;
    font-style: italic;
    color: #444;
    margin-bottom: 25px;
}}

.callout-box {{
    background: #eef4fc;
    border-left: 4px solid #1e6091;
    padding: 15px 20px;
    margin: 20px 0;
    font-size: 11pt;
}}

.title-sheet {{
    text-align: center;
    min-height: 900px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 60px 20px;
    page-break-after: always;
    break-after: page;
}}

.sig-block {{
    margin-top: 60px;
    display: flex;
    justify-content: space-between;
}}

.btn-print {{
    position: fixed;
    top: 20px;
    right: 20px;
    background: #1a73e8;
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    z-index: 1000;
}}
.btn-print:hover {{
    background: #1558b0;
}}
</style>
</head>
<body>

<button class="btn-print no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>

<div class="page-container">

<!-- SPECIMEN A: TITLE SHEET -->
<div class="title-sheet">
    <div>
        <div style="font-size: 13pt; letter-spacing: 2px; color: #555; margin-bottom: 25px;">B.TECH PROJECT REPORT</div>
        <h1 class="report-title">AI-Based Kinetic Modeling of Molten-Metal Bubble Column Reactors for Green Hydrogen Production via Methane Pyrolysis</h1>
    </div>
    
    <div style="margin: 40px 0;">
        <p style="text-align:center; font-size: 13pt;">A Project Report submitted in partial fulfillment of the<br>requirements for the degree of</p>
        <p style="text-align:center; font-size: 15pt; font-weight: bold; margin-top: 15px;">BACHELOR OF TECHNOLOGY</p>
    </div>

    <div>
        <p style="text-align:center; font-size: 12pt;">By</p>
        <p style="text-align:center; font-size: 15pt; font-weight: bold;">SIDDHARTH PATEL</p>
        <p style="text-align:center; font-size: 12pt;">(Roll No. [Your Roll Number])</p>
    </div>

    <div style="margin-top: 50px;">
        <p style="text-align:center; font-size: 13pt; font-weight: bold; margin-bottom: 4px;">DEPARTMENT OF CHEMICAL ENGINEERING</p>
        <p style="text-align:center; font-size: 14pt; font-weight: bold; margin-bottom: 4px;">INDIAN INSTITUTE OF TECHNOLOGY GUWAHATI</p>
        <p style="text-align:center; font-size: 12pt;">NOVEMBER 2026</p>
    </div>
</div>

<!-- SPECIMEN B: CERTIFICATE -->
<div class="page-break" style="padding-top: 40px;">
    <h2 style="text-align: center; text-transform: uppercase;">Certificate</h2>
    <div style="margin-top: 50px; font-size: 12pt;">
        <p>It is certified that the work contained in the project report titled <strong>"AI-Based Kinetic Modeling of Molten-Metal Bubble Column Reactors for Green Hydrogen Production via Methane Pyrolysis,"</strong> by <strong>Siddharth Patel</strong> (Roll No. [Your Roll Number]), has been carried out under my supervision and that this work has not been submitted elsewhere for a degree.</p>
    </div>
    <div style="margin-top: 120px; float: right; text-align: right;">
        <div style="border-top: 1px solid #333; width: 250px; margin-bottom: 8px;"></div>
        <strong>[Supervisor Name]</strong><br>
        Department of Chemical Engineering<br>
        Indian Institute of Technology Guwahati<br>
        Date: _______________
    </div>
    <div style="clear: both;"></div>
</div>

<!-- SPECIMEN C: DECLARATION -->
<div class="page-break" style="padding-top: 40px;">
    <h2 style="text-align: center; text-transform: uppercase;">Declaration of Academic Honesty and Integrity</h2>
    <div style="margin-top: 50px; font-size: 12pt;">
        <p>I declare that this written submission represents my ideas in my own words and where others' ideas or words have been included, I have adequately cited and referenced the original sources. I also declare that I have adhered to all principles of academic honesty and integrity and have not misrepresented or fabricated or falsified any idea/data/fact/source in my submission. I understand that any violation of the above will be cause for disciplinary action by the Institute and can also evoke penal action from the sources which have thus not been properly cited or from whom proper permission has not been taken when needed.</p>
    </div>
    <div style="margin-top: 120px; float: right; text-align: right;">
        <div style="border-top: 1px solid #333; width: 250px; margin-bottom: 8px;"></div>
        (Signature)<br>
        <strong>Siddharth Patel</strong><br>
        Roll No. [Your Roll Number]<br>
        Date: _______________
    </div>
    <div style="clear: both;"></div>
</div>

<!-- SPECIMEN D: APPROVAL -->
<div class="page-break" style="padding-top: 40px;">
    <h2 style="text-align: center; text-transform: uppercase;">Report Approval for B.Tech Project</h2>
    <div style="margin-top: 40px; font-size: 12pt;">
        <p>This project report entitled <strong>"AI-Based Kinetic Modeling of Molten-Metal Bubble Column Reactors for Green Hydrogen Production via Methane Pyrolysis"</strong> by <strong>Siddharth Patel</strong> is approved for the degree of Bachelor of Technology in Chemical Engineering.</p>
    </div>
    <div style="margin-top: 60px;">
        <table style="border: none;">
            <tr style="background: transparent; border: none;">
                <td style="border: none; width: 50%;">
                    <strong>Examiners:</strong><br><br><br>
                    ___________________________<br><br><br>
                    ___________________________
                </td>
                <td style="border: none; width: 50%;">
                    <strong>Supervisor:</strong><br><br><br>
                    ___________________________<br><br><br>
                    <strong>Chairman:</strong><br><br><br>
                    ___________________________
                </td>
            </tr>
        </table>
        <p style="margin-top: 40px;">Date: _______________<br>Place: IIT Guwahati</p>
    </div>
</div>

<!-- MAIN CONTENT FROM MARKDOWN -->
<div class="page-break">
"""

    # Simple regex conversion of Markdown to HTML
    body_md = md_text

    # Convert sections 1 through 11 into clean chapter titles
    def replace_h2(match):
        title = match.group(1)
        return f'<h1 class="chapter-title">{title}</h1>'
    
    body_md = re.sub(r'^##\s+(.+)$', replace_h2, body_md, flags=re.MULTILINE)
    body_md = re.sub(r'^###\s+(.+)$', r'<h2>\1</h2>', body_md, flags=re.MULTILINE)
    body_md = re.sub(r'^####\s+(.+)$', r'<h3>\1</h3>', body_md, flags=re.MULTILINE)

    # Convert image embeds
    def replace_img(match):
        alt = match.group(1)
        src = match.group(2)
        # Fix relative image paths if necessary
        return f'<img src="../{src}" alt="{alt}"><div class="figure-caption">{alt}</div>'
    
    body_md = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, body_md)

    # Convert markdown tables
    lines = body_md.split("\n")
    in_table = False
    html_lines = []
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if not in_table:
                in_table = True
                html_lines.append("<table>")
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            if all(set(c).issubset({"-", ":", " "}) for c in cells):
                continue
            tag = "th" if "<table>" in html_lines[-1] else "td"
            row = "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            html_lines.append(row)
        else:
            if in_table:
                in_table = False
                html_lines.append("</table>")
            html_lines.append(line)
    if in_table:
        html_lines.append("</table>")

    body_html = "\n".join(html_lines)

    # Wrap standalone paragraphs
    body_html = re.sub(r'\n\n([^<\n][^\n]+)\n\n', r'\n\n<p>\1</p>\n\n', body_html)

    full_html = html_template + body_html + """
</div>
</div>
</body>
</html>
"""

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated printable HTML: {HTML_OUT}")

if __name__ == "__main__":
    build_printable_html()

