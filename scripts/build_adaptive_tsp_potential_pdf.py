"""Typeset the proof with LaTeX and assemble its PDF with ReportLab/pypdf."""
from io import BytesIO
from pathlib import Path
import subprocess

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis/adaptive_tsp_exponential_potential_proof.tex"
BUILD = ROOT / "tmp/pdfs/adaptive_tsp_potential"
OUTPUT = ROOT / "output/pdf/adaptive_tsp_exponential_potential_proof.pdf"
BUILD.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

for _ in range(2):
    result = subprocess.run(
        [
            "/Library/TeX/texbin/pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={BUILD}",
            str(SOURCE),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)

body = PdfReader(BUILD / f"{SOURCE.stem}.pdf")
shell_stream = BytesIO()
shell = canvas.Canvas(shell_stream, pageCompression=1)
total = len(body.pages)
for number, page in enumerate(body.pages, start=1):
    width, height = map(float, page.mediabox[2:])
    shell.setPageSize((width, height))
    shell.setStrokeColor(HexColor("#CDD5DD"))
    shell.setLineWidth(0.4)
    shell.line(73.7, 49, width - 73.7, 49)
    shell.setFillColor(HexColor("#556477"))
    shell.setFont("Helvetica", 8)
    shell.drawString(73.7, 35, "ADAPTIVE MULTIPLICITY BALANCING")
    shell.drawRightString(width - 73.7, 35, f"{number} / {total}")
    shell.showPage()
shell.save()
shell_stream.seek(0)

shell_reader = PdfReader(shell_stream)
writer = PdfWriter()
for base_page, content_page in zip(shell_reader.pages, body.pages):
    base_page.merge_page(content_page)
    writer.add_page(base_page)
writer.add_metadata(
    {
        "/Title": "A Short Potential Proof for Adaptive Multiplicity Balancing",
        "/Subject": "Causal balancing with one, two, or three copies and its adaptive-TSP application",
        "/Keywords": "adaptive TSP, discrepancy, exponential potential, remaining matrix",
    }
)
with OUTPUT.open("wb") as stream:
    writer.write(stream)
print(f"Created {OUTPUT} ({total} pages)")
