"""Compile selected report figures standalone (Be Vietnam Pro text, report colours) to fig_png/<name>.png at 300 dpi."""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src3d = open(os.path.join(HERE, "fig/pgf-3d-surfaces.tex"), encoding="utf-8").read()
B, E = chr(92) + "begin{tikzpicture}", chr(92) + "end{tikzpicture}"
pics = []
i = 0
while src3d.find(B, i) >= 0:
    a = src3d.find(B, i); b = src3d.find(E, a) + len(E); pics.append(src3d[a:b]); i = b
FIGS = {"vig3d": pics[0], "dc3d": pics[1], "rgbcube": r"\input{fig/pgf-rgbcube}", "timeline": r"\input{fig/tikz-timeline}"}
PRE = r"""\documentclass[12pt,border=4pt]{standalone}
\usepackage{fontspec}
\setmainfont{BeVietnamPro}[Path=D:/IOT_Tools/downloads/fonts/,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-SemiBold]
\usepackage{amsmath}
\usepackage{unicode-math}\setmathfont{Cambria Math}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric,calc,fit,backgrounds,decorations.pathreplacing,shadows}
\usepackage{tikz-3dplot}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots,fillbetween}
\definecolor{cXanh}{HTML}{1F5A96}\definecolor{cCam}{HTML}{E07B24}\definecolor{cLuc}{HTML}{2E8B57}
\definecolor{cHong}{HTML}{C2397F}\definecolor{cTim}{HTML}{6B4FA0}\definecolor{cXam}{HTML}{5B6770}\definecolor{cVang}{HTML}{D8A51D}
\pgfplotsset{baocao/.style={tick label style={font=\scriptsize}, scaled ticks=false, /pgf/number format/fixed,
  label style={font=\small}, legend style={font=\scriptsize,draw=none,fill=none},
  /pgf/number format/use comma, /pgf/number format/1000 sep={\,}, axis line style={cXam}, major tick style={cXam},
  ymajorgrids, grid style={cXam!20}}}
\pgfplotsset{colormap={baocao}{rgb255=(31,90,150) rgb255=(107,79,160) rgb255=(194,57,127) rgb255=(224,123,36) rgb255=(216,165,29)}}
\pgfkeys{/pgf/number format/assume math mode=true}
\newcommand{\ci}[2]{[#1;\,#2]}\newcommand{\so}[1]{#1}
\begin{document}
"""
os.makedirs(os.path.join(HERE, "fig_png"), exist_ok=True)
for name, body in FIGS.items():
    with open(os.path.join(HERE, f"f_{name}.tex"), "w", encoding="utf-8") as f:
        f.write((PRE.replace("[12pt,", "[10pt,") if name == "timeline" else PRE) + body + "\n\end{document}\n")
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"f_{name}.tex"], cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(name, r.stdout[-2500:]); sys.exit(1)
    subprocess.run(["pdftocairo", "-png", "-r", "300", "-singlefile", "-transp", f"f_{name}.pdf", f"fig_png/{name}"], cwd=HERE, check=True)
    print("ok", name)
