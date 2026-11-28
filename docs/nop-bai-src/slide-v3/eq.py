import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "stix"
fig = plt.figure(figsize=(8.3, 0.73), dpi=300)
fig.text(0.5, 0.5, r"$t(x)=\max\left(0.4,\ 1-\omega\,\frac{\mathrm{dc}(x)}{\bar{A}}\right),\qquad J_c=\frac{I_c-A_c}{t(x)}+A_c$",
         ha="center", va="center", fontsize=27, color="#1b2433")
fig.savefig(sys.argv[1], transparent=True)
