"""Render every snippet in eqlist.EQ to eq/<name>.png (300 dpi, transparent), one standalone page each."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eqlist import COLOR, EQ, INK  # noqa: E402

FONTDIR = "D:/IOT_Tools/downloads/fonts/"
PRE = r"""\documentclass[border=1.5pt,multi=eqn]{standalone}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{BeVietnamPro}[Path=%s,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-SemiBold]
\setmathfont{Cambria Math}
\setmathfont{Cambria Math}[range={"2013}]
\newenvironment{eqn}[1]{\fontsize{18}{22}\selectfont\color[HTML]{#1}$\displaystyle}{$}
\begin{document}
""" % FONTDIR

names = list(EQ)
body = "".join("\\begin{eqn}{%s}%s\\end{eqn}\n" % (COLOR.get(n, INK), EQ[n]) for n in names)
os.makedirs(os.path.join(HERE, "eq"), exist_ok=True)
with open(os.path.join(HERE, "eqs.tex"), "w", encoding="utf-8") as f:
    f.write(PRE + body + "\\end{document}\n")
r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "eqs.tex"], cwd=HERE,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode:
    print(r.stdout[-3000:]); sys.exit("xelatex failed")
subprocess.run(["pdftocairo", "-png", "-r", "300", "-transp", "eqs.pdf", "eq/p"], cwd=HERE, check=True)
pages = sorted(p for p in os.listdir(os.path.join(HERE, "eq")) if p.startswith("p-"))
assert len(pages) == len(names), (len(pages), len(names))
for p, n in zip(pages, names):
    os.replace(os.path.join(HERE, "eq", p), os.path.join(HERE, "eq", n + ".png"))
print("rendered", len(names))
