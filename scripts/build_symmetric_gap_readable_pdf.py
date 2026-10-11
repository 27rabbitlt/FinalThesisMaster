#!/usr/bin/env python3
"""Typeset the Chinese readable proof with native XeLaTeX mathematics."""

from pathlib import Path
import importlib.util
import re
import subprocess
import argparse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis/symmetric_gap_lower_bound_readable_zh.md"
BUILD = ROOT / "tmp/pdfs/symmetric-gap-readable"
OUTPUT = ROOT / "output/pdf/symmetric_gap_lower_bound_readable_zh.pdf"

spec = importlib.util.spec_from_file_location(
    "research_markdown", ROOT / "scripts/markdown_math_to_tex.py"
)
markdown = importlib.util.module_from_spec(spec)
spec.loader.exec_module(markdown)


def inline(text):
    """Keep nested emphasis and native math intact without placeholder scopes."""
    pattern = re.compile(r"\*\*.+?\*\*|\\\(.+?\\\)|`[^`]+`|\[[^\]]+\]\([^)]+\)", re.DOTALL)
    output = []
    previous = 0
    for match in pattern.finditer(text):
        output.append(markdown.escape_text(text[previous:match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            output.append(r"\textbf{" + inline(token[2:-2]) + "}")
        elif token.startswith(r"\("):
            output.append(token)
        elif token.startswith("`"):
            output.append(r"\nolinkurl{" + token[1:-1] + "}")
        else:
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            label, target = link.groups()
            output.append(r"\href{" + markdown.escape_text(target) + "}{" + inline(label) + "}")
        previous = match.end()
    output.append(markdown.escape_text(text[previous:]))
    return "".join(output)


def table(lines):
    rows = [[s.strip() for s in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", s) for s in r)]
    columns = len(rows[0])
    specs = r">{\raggedright\arraybackslash}X" + r">{\centering\arraybackslash}X" * (columns - 1)
    result = [
        r"\begin{center}",
        r"\begingroup\small",
        r"\renewcommand{\arraystretch}{1.45}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabularx}{\linewidth}{" + specs + "}",
        r"\toprule\rowcolor{Pale}",
    ]
    for i, row in enumerate(rows):
        cells = [markdown.inline(cell) for cell in row]
        if i == 0:
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        result.append(" & ".join(cells) + r" \\")
        if i == 0:
            result.append(r"\midrule")
    result.extend([r"\bottomrule", r"\end{tabularx}", r"\endgroup", r"\end{center}"])
    return "\n".join(result)


def main():
    global SOURCE, BUILD, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned", action="store_true", help="Build the 2.2215 parameter-tuning supplement")
    args = parser.parse_args()
    if args.tuned:
        SOURCE = ROOT / "thesis/symmetric_tuned_layer_lower_bound_zh.md"
        BUILD = ROOT / "tmp/pdfs/symmetric-gap-tuned"
        OUTPUT = ROOT / "output/pdf/symmetric_tuned_layer_lower_bound_zh.pdf"
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    markdown.inline = inline
    markdown.parse_table = table
    source = re.sub(r"^---\s*$", "", SOURCE.read_text(), flags=re.MULTILINE)
    title, body, _ = markdown.parse_markdown(source)
    body = body.replace(r"\phantomsection", r"\Needspace{6\baselineskip}" + "\n" + r"\phantomsection")
    # Keep enough room for the technical heading without creating a nearly empty page.
    body = body.replace(
        "\\Needspace{6\\baselineskip}\n\\phantomsection\n\\section*{技术细节 A",
        "\\Needspace{12\\baselineskip}\n\\phantomsection\n\\section*{技术细节 A",
    )
    diagram = r"""\begin{center}
\begin{tikzpicture}[box/.style={rounded corners=2pt,inner xsep=12pt,inner ysep=9pt,font=\ttfamily\large}]
\node[box,fill=Pale,text=Ink] (left) at (-4,0) {M M M};
\node[box,fill=Accent!14,text=Accent] (middle) at (0,0) {H H H H H};
\node[box,fill=Pale,text=Ink] (right) at (4,0) {M M M};
\draw[Accent!55,line width=.7pt] (left.east)--(middle.west);
\draw[Accent!55,line width=.7pt] (middle.east)--(right.west);
\end{tikzpicture}
\end{center}"""
    body = re.sub(
        r"\\begin\{Verbatim\}\[[^\]]*\]\s*M M M\s*\|\s*H H H H H\s*\|\s*M M M\s*\\end\{Verbatim\}",
        lambda _: diagram,
        body,
    )
    preamble = r"""\documentclass[UTF8,a4paper,11pt,fontset=fandol]{ctexart}
\usepackage[left=24mm,right=24mm,top=23mm,bottom=23mm,headheight=16pt]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Avenir Next}
\setmonofont{Menlo}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{newtxmath}
\usepackage{microtype}
\usepackage{booktabs,array,tabularx,longtable}
\usepackage[table]{xcolor}
\usepackage{enumitem,needspace,fancyhdr,lastpage}
\usepackage{fvextra,xurl}
\usepackage[most]{tcolorbox}
\usepackage{tikz}
\usepackage{hyperref,bookmark}
\definecolor{Ink}{HTML}{203B50}
\definecolor{Accent}{HTML}{276A70}
\definecolor{Pale}{HTML}{EEF3F5}
\definecolor{LinkBlue}{HTML}{276A70}
\hypersetup{colorlinks=true,linkcolor=Accent,urlcolor=Accent,
pdftitle={从组的两端理解对称 adaptivity gap 的 2.18 下界},
pdfauthor={},bookmarksopen=true,bookmarksnumbered=false}
\ctexset{section={format=\Large\bfseries\sffamily\color{Ink},
beforeskip=1.35em,afterskip=.65em},
subsection={format=\large\bfseries\sffamily\color{Ink}}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{.48em}
\linespread{1.18}
\setlength{\emergencystretch}{3em}
\setlength{\abovedisplayskip}{9pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{9pt plus 2pt minus 2pt}
\setlist{leftmargin=1.6em,itemsep=.22em,topsep=.25em,parsep=0pt}
\setcounter{tocdepth}{1}
\allowdisplaybreaks[1]
\widowpenalty=10000
\clubpenalty=10000
\newtcolorbox{ResearchQuote}{enhanced,breakable,colback=Pale,colframe=Accent,
boxrule=0pt,leftrule=2pt,arc=0pt,left=9pt,right=9pt,top=6pt,bottom=6pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily\color{Ink}对称随机 TSP}
\fancyhead[R]{\small\sffamily\color{Ink}2.18 下界 · 阅读版}
\fancyfoot[C]{\small\sffamily\thepage\,/\,\pageref*{LastPage}}
\renewcommand{\headrulewidth}{.3pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\small\sffamily\thepage\,/\,\pageref*{LastPage}}\renewcommand{\headrulewidth}{0pt}}
\begin{document}
\thispagestyle{plain}
{\sffamily\color{Accent}\small 直观机制与完整证明}\par
\vspace{4mm}
{\sffamily\bfseries\color{Ink}\fontsize{23}{32}\selectfont
从“组的两端”理解\\对称 adaptivity gap 的 2.18 下界}\par
\vspace{4mm}
{\color{Accent}\rule{42mm}{1.2pt}}\par
{\small 2026 年 9 月 10 日}\par
\vspace{3mm}
\begingroup
\small\setlength{\parskip}{0pt}
\tableofcontents
\endgroup
\clearpage
"""
    if args.tuned:
        preamble = preamble.replace(
            "从组的两端理解对称 adaptivity gap 的 2.18 下界",
            "调整距离之后：对称 adaptivity gap 严格超过 2.2215",
        ).replace("2.18 下界 · 阅读版", "2.2215 下界 · 参数调整").replace(
            "直观机制与完整证明", "阅读版补充 · 参数调整与严格证书",
        ).replace(
            "从“组的两端”理解\\\\对称 adaptivity gap 的 2.18 下界",
            "调整距离之后\\\\同一两层构造，严格超过 2.2215",
        ).replace(r"\linespread{1.18}", r"\linespread{1.12}").replace(
            r"\setlength{\parskip}{.48em}", r"\setlength{\parskip}{.40em}",
        )
    tex = BUILD / (OUTPUT.stem + ".tex")
    tex.write_text(preamble + "\n" + body + "\n\\end{document}\n")
    command = [
        "/Library/TeX/texbin/xelatex", "-interaction=nonstopmode", "-halt-on-error",
        "-file-line-error", f"-output-directory={BUILD}", str(tex),
    ]
    for run in range(3):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (BUILD / f"compile-{run + 1}.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout[-9000:])
            raise SystemExit(result.returncode)
    OUTPUT.write_bytes(tex.with_suffix(".pdf").read_bytes())
    print(OUTPUT)


if __name__ == "__main__":
    main()
