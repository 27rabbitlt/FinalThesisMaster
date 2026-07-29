#!/usr/bin/env python3
"""Convert the project's research-note Markdown subset to polished XeLaTeX.

The notes already contain native LaTeX math delimiters.  This converter keeps
those expressions intact while translating headings, prose, lists, quotes,
inline code, emphasis, and pipe tables.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TOKEN_RE = re.compile(r"@@PROTECTED(\d+)@@")


def escape_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def inline(text: str) -> str:
    protected: list[str] = []

    def keep(value: str) -> str:
        protected.append(value)
        return f"@@PROTECTED{len(protected) - 1}@@"

    # Preserve native inline mathematics.
    text = re.sub(r"\\\(.*?\\\)", lambda m: keep(m.group(0)), text)

    # Convert Markdown links and code before ordinary escaping.
    def link_repl(match: re.Match[str]) -> str:
        label = inline(match.group(1))
        target = match.group(2)
        if target.startswith(("http://", "https://")):
            return keep(r"\href{" + escape_text(target) + "}{" + label + "}")
        return keep(r"\textcolor{LinkBlue}{" + label + "}")

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)
    text = re.sub(
        r"`([^`]+)`",
        lambda m: keep(r"\nolinkurl{" + m.group(1) + "}"),
        text,
    )

    # Bold spans can contain already protected math/code tokens.
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: keep(r"\textbf{" + inline(m.group(1)) + "}"),
        text,
    )
    text = escape_text(text)

    def restore(match: re.Match[str]) -> str:
        return protected[int(match.group(1))]

    previous = None
    while previous != text:
        previous = text
        text = TOKEN_RE.sub(restore, text)
    return text


def parse_table(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", c) for c in rows[1]):
        rows.pop(1)
    columns = max(len(row) for row in rows)
    if columns == 4:
        spec = (
            r">{\raggedright\arraybackslash}p{0.15\linewidth}"
            r">{\raggedright\arraybackslash}p{0.27\linewidth}"
            r">{\raggedright\arraybackslash}p{0.28\linewidth}"
            r">{\raggedright\arraybackslash}p{0.14\linewidth}"
        )
    else:
        width = max(0.10, 0.78 / columns)
        spec = "".join(
            rf">{{\raggedright\arraybackslash}}p{{{width:.3f}\textwidth}}"
            for _ in range(columns)
        )
    output = [
        r"\begin{landscape}" if columns == 4 else "",
        "" if columns == 4 else r"\begin{center}",
        r"\begingroup\small" if columns == 4 else r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        rf"\begin{{longtable}}{{{spec}}}",
        r"\rowcolor{TableHead}",
    ]
    for index, row in enumerate(rows):
        row += [""] * (columns - len(row))
        formatted = [inline(cell) for cell in row]
        if index == 0:
            formatted = [r"\textbf{" + cell + "}" for cell in formatted]
        output.append(" & ".join(formatted) + r" \\")
        output.append(r"\hline")
        if index == 0:
            output.extend(
                [
                    r"\endfirsthead",
                    r"\rowcolor{TableHead}",
                    " & ".join(formatted) + r" \\",
                    r"\hline",
                    r"\endhead",
                ]
            )
    output.extend(
        [
            r"\end{longtable}",
            r"\endgroup",
            "" if columns == 4 else r"\end{center}",
            r"\end{landscape}" if columns == 4 else "",
        ]
    )
    return "\n".join(output)


def parse_markdown(source: str) -> tuple[str, str, int]:
    lines = source.splitlines()
    title = "Research Note"
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]

    output: list[str] = []
    section_count = 0
    index = 0

    def is_boundary(line: str) -> bool:
        stripped = line.strip()
        return (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith(r"\[")
            or stripped.startswith("```")
            or stripped.startswith(">")
            or stripped.startswith("|")
            or re.match(r"^\s*[-*+]\s+", line) is not None
            or re.match(r"^\s*\d+\.\s+", line) is not None
        )

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        heading = re.match(r"^(#{2,4})\s+(.+)$", line)
        if heading:
            depth = len(heading.group(1))
            command = {2: "section", 3: "subsection", 4: "subsubsection"}[depth]
            toc_level = {2: "section", 3: "subsection", 4: "subsubsection"}[depth]
            if depth == 2:
                section_count += 1
            heading_text = inline(heading.group(2))
            output.append(
                rf"\phantomsection"
                + "\n"
                + rf"\{command}*{{{heading_text}}}"
                + "\n"
                + rf"\addcontentsline{{toc}}{{{toc_level}}}{{{heading_text}}}"
            )
            index += 1
            continue

        if stripped.startswith(r"\["):
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip().endswith(r"\]"):
                    index += 1
                    break
                index += 1
            raw_block = "\n".join(block)
            tags = re.findall(r"\\tag\{([^}]+)\}", raw_block)
            if tags:
                inner = "\n".join(block[1:-1])
                inner = re.sub(r"\s*\\tag\{[^}]+\}", "", inner)
                output.append(
                    "\\begin{equation}\n"
                    + inner
                    + "\n"
                    + rf"\tag{{{tags[-1]}}}"
                    + "\n\\end{equation}"
                )
            else:
                output.append(raw_block)
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            output.extend(
                [
                    r"\begin{Verbatim}[fontsize=\small,breaklines=true]",
                    "\n".join(code),
                    r"\end{Verbatim}",
                ]
            )
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            output.append(parse_table(table_lines))
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            output.extend(
                [
                    r"\begin{ResearchQuote}",
                    inline(" ".join(quote_lines)),
                    r"\end{ResearchQuote}",
                ]
            )
            continue

        list_match = re.match(r"^\s*([-*+]|\d+\.)\s+(.+)$", line)
        if list_match:
            ordered = list_match.group(1)[0].isdigit()
            environment = "enumerate" if ordered else "itemize"
            output.append(rf"\begin{{{environment}}}")
            while index < len(lines):
                current = re.match(r"^\s*([-*+]|\d+\.)\s+(.+)$", lines[index])
                if not current or current.group(1)[0].isdigit() != ordered:
                    break
                item_parts = [current.group(2).strip()]
                index += 1
                while index < len(lines):
                    continuation = lines[index]
                    if not continuation.strip() or is_boundary(continuation):
                        break
                    item_parts.append(continuation.strip())
                    index += 1
                output.append(r"\item " + inline(" ".join(item_parts)))
                if index < len(lines) and not lines[index].strip():
                    # A blank line ends the current list in this Markdown subset.
                    break
            output.append(rf"\end{{{environment}}}")
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines) and not is_boundary(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        output.append(inline(" ".join(paragraph)) + "\n")

    return title, "\n\n".join(output), section_count


def make_document(title: str, body: str, source_name: str, section_count: int) -> str:
    formatted_title = inline(title)
    if len(title) > 52:
        title_font = r"\fontsize{20}{24}\selectfont"
    elif len(title) > 32:
        title_font = r"\fontsize{21}{25}\selectfont"
    else:
        title_font = r"\Huge"
    toc = (
        r"""
\begin{ResearchContents}
\tableofcontents
\end{ResearchContents}
\clearpage
"""
        if section_count >= 3
        else ""
    )
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=24mm,headheight=15pt]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\setsansfont{{Avenir Next}}
\setmonofont{{Menlo}}
\usepackage{{microtype}}
\usepackage{{newtxmath}}
\usepackage{{mathtools}}
\usepackage{{booktabs,longtable,array,tabularx}}
\usepackage{{pdflscape}}
\usepackage[table]{{xcolor}}
\usepackage{{enumitem}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{xurl}}
\usepackage{{hyperref}}
\usepackage{{bookmark}}
\usepackage{{fancyvrb}}
\usepackage[most]{{tcolorbox}}

\definecolor{{Navy}}{{HTML}}{{18324A}}
\definecolor{{Accent}}{{HTML}}{{C56A3A}}
\definecolor{{LinkBlue}}{{HTML}}{{2D6A8A}}
\definecolor{{SoftBlue}}{{HTML}}{{EAF1F5}}
\definecolor{{TableHead}}{{HTML}}{{DCE8EF}}
\definecolor{{BodyGray}}{{HTML}}{{26343D}}

\hypersetup{{
  colorlinks=true,
  linkcolor=LinkBlue,
  urlcolor=LinkBlue,
  citecolor=LinkBlue,
  pdftitle={{{escape_text(title)}}},
  pdfauthor={{Final Thesis Master Project}}
}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.65em}}
\setlength{{\emergencystretch}}{{3em}}
\setlist{{leftmargin=1.6em,itemsep=0.3em,topsep=0.25em}}
\setcounter{{tocdepth}}{{2}}
\renewcommand{{\arraystretch}}{{1.15}}
\allowdisplaybreaks

\titleformat{{\section}}
  {{\Large\bfseries\sffamily\color{{Navy}}}}
  {{\thesection}}{{0.65em}}{{}}
  [\vspace{{-0.25em}}\color{{Accent}}\titlerule]
\titleformat{{\subsection}}
  {{\large\bfseries\sffamily\color{{Navy}}}}
  {{\thesubsection}}{{0.6em}}{{}}
\titleformat{{\subsubsection}}
  {{\normalsize\bfseries\sffamily\color{{LinkBlue}}}}
  {{\thesubsubsection}}{{0.55em}}{{}}

\newtcolorbox{{ResearchQuote}}{{
  enhanced,
  breakable,
  colback=SoftBlue,
  colframe=LinkBlue,
  boxrule=0pt,
  leftrule=2.2pt,
  arc=0pt,
  left=8pt,right=8pt,top=5pt,bottom=5pt
}}
\newenvironment{{ResearchContents}}
  {{\begin{{tcolorbox}}[breakable,colback=SoftBlue,colframe=SoftBlue,arc=2pt]}}
  {{\end{{tcolorbox}}}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\sffamily\color{{Navy}} Asymmetric stochastic TSP}}
\fancyhead[R]{{\small\sffamily\color{{Navy}}\detokenize{{{source_name}}}}}
\fancyfoot[C]{{\small\sffamily\color{{BodyGray}}\thepage}}
\renewcommand{{\headrulewidth}}{{0.3pt}}
\renewcommand{{\footrulewidth}}{{0pt}}

\begin{{document}}
\color{{BodyGray}}
\begin{{titlepage}}
  \thispagestyle{{empty}}
  \vspace*{{22mm}}
  {{\sffamily\small\bfseries\color{{Accent}}\MakeUppercase{{Research note}}}}\par
  \vspace{{8mm}}
  {{\sffamily{title_font}\bfseries\color{{Navy}} {formatted_title}}}\par
  \vspace{{6mm}}
  {{\color{{Accent}}\rule{{42mm}}{{1.4pt}}}}\par
  \vspace{{11mm}}
  {{\Large\color{{BodyGray}} Asymmetric stochastic TSP\\
  clairvoyance-gap investigation}}\par
  \vfill
  \begin{{tcolorbox}}[
    colback=SoftBlue,colframe=SoftBlue,arc=2pt,
    left=10pt,right=10pt,top=8pt,bottom=8pt]
    \sffamily
    \textbf{{Source}}\quad \detokenize{{{source_name}}}\\[3pt]
    \textbf{{Prepared}}\quad 25 July 2026\\[3pt]
    \textbf{{Format}}\quad Typeset mathematical PDF
  \end{{tcolorbox}}
  \vspace{{12mm}}
\end{{titlepage}}
{toc}
{body}
\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    title, body, section_count = parse_markdown(args.source.read_text())
    document = make_document(title, body, args.source.name, section_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)


if __name__ == "__main__":
    main()
