"""Formula snippets for the deck. name -> (latex, color). Rendered by mkeq.py to eq/<name>.png at 300 dpi.
Every formula follows the code in D:/XLA/xla (see the investigation notes); textbook forms are marked *_book."""
INK, WHITE, PINK = "0E2238", "FFFFFF", "B4232F"

EQ = {
    # --- §3 theory
    "haze": r"I(x)=J(x)\,t(x)+A\,\bigl(1-t(x)\bigr)",
    "haze_book": r"t(x)=e^{-\beta\,d(x)}",
    "haze_d2": r"t=\operatorname{clip}\Bigl(\dfrac{A-d^{*}}{A-d_J};\;0{,}2;\;1\Bigr)",
    "dark_meas": r"D=\dfrac{1}{N}\sum_{x}\,\min_{y\in\Omega_{15}(x)}\,\min_{c}\,I^{c}(y)",
    "vig": r"G(r)=\cos^{4}\Bigl(\arctan\dfrac{r}{f}\Bigr)=\Bigl(1+\dfrac{r^{2}}{f^{2}}\Bigr)^{-2}",
    "vig_obs": r"I(x)=J(x)\cdot G\bigl(r(x)\bigr)",
    "vig_fit": r"f=\dfrac{r_c}{\tan\bigl(\arccos V^{1/4}\bigr)}=1016\ \text{px}",
    "dct": r"\begin{aligned}F(u,v)&=\tfrac14\,C(u)\,C(v)\sum_{x=0}^{7}\sum_{y=0}^{7}f(x,y)\,\phi_u(x)\,\phi_v(y)\\ \phi_u(x)&=\cos\dfrac{(2x+1)u\pi}{16},\quad C(0)=\tfrac{1}{\sqrt{2}},\ \ C(k)=1\end{aligned}",
    "dct_c": r"C(0)=1/\sqrt{2},\quad C(k)=1\ \ (k>0)",
    "quant": r"F_q=\operatorname{round}\dfrac{F(u,v)}{Q(u,v)},\qquad Q=\Bigl\lfloor\dfrac{75\,Q_K+50}{100}\Bigr\rfloor",
    "block": r"B=\tfrac12\Bigl(\dfrac{\overline{|\Delta_x Y|}_{\text{biên}}}{\overline{|\Delta_x Y|}_{\text{trong}}}+\dfrac{\overline{|\Delta_y Y|}_{\text{biên}}}{\overline{|\Delta_y Y|}_{\text{trong}}}\Bigr)",
    "pinkrule": r"P(x)=\bigl[R-G>25\bigr]\wedge\bigl[B-G>25\bigr]",
    "tone": r"O_c=\operatorname{clip}\bigl(k\,g_c\,c_c\,\min(e\,I_c/g_c,\;1)\bigr)",
    "tone_v": r"g_R=2{,}13,\quad g_B=1{,}812,\quad g_G=1",
    # --- §5 D2 nodes + objective
    "d2a": r"I=J\,t+A(1-t)",
    "d2b": r"(G_\sigma\ast I)\,\cos^{4}\theta",
    "d2c": r"\min(e\,I_c/g_c,\,1)",
    "d2d": r"I+\sigma(Y)\,n",
    "d2f": r"\operatorname{round}(F/Q_{12})",
    "obj_w": r"\mathcal{L}=\dfrac{1}{|K|}\sum_{k\in K}\dfrac{W_1(\text{thật}_k,\ \text{D2}_k)}{\operatorname{IQR}(\text{thật}_k)}",
    "stat": r"\mathcal{L}=\dfrac{1}{|K|}\sum_{k\in K}\dfrac{W_1(\text{thật}_k,\ \text{giả}_k)}{\operatorname{IQR}(\text{thật}_k)},\quad W_1=\int\bigl|F_a-F_b\bigr|",
    # --- §6 realism
    "hp": r"r=I-G_{\sigma=2}\ast I",
    "c2st": r"\mathrm{acc}=\dfrac{1}{n}\sum_{i}\bigl[\hat y_i=y_i\bigr]\ \to\ 0{,}5",
    "kid": r"\begin{aligned}\widehat{\mathrm{MMD}}^{2}&=\tfrac{1}{m(m-1)}\sum_{i\neq j}k(x_i,x_j)+\tfrac{1}{n(n-1)}\sum_{i\neq j}k(y_i,y_j)\\&\quad-\tfrac{2}{mn}\sum_{i,j}k(x_i,y_j)\end{aligned}",
    "kernel": r"k(x,y)=\Bigl(\dfrac{x^{\top}y}{d}+1\Bigr)^{3},\quad d=1024",
    # --- §7 YOLO11s
    "yloss": r"\mathcal{L}=7{,}5\,\mathcal{L}_{\text{box}}+0{,}5\,\mathcal{L}_{\text{cls}}+1{,}5\,\mathcal{L}_{\text{DFL}}",
    "ciou": r"\begin{aligned}\mathcal{L}_{\text{box}}&=1-\mathrm{IoU}+\dfrac{\rho^{2}(b,b^{gt})}{c^{2}}+\alpha v\\ v&=\dfrac{4}{\pi^{2}}\Bigl(\arctan\dfrac{w^{gt}}{h^{gt}}-\arctan\dfrac{w}{h}\Bigr)^{2}\end{aligned}",
    "dfl": r"\begin{aligned}\mathcal{L}_{\text{DFL}}&=-\bigl[(y_{r}-y)\log S_{l}+(y-y_{l})\log S_{r}\bigr]\\ \hat d&=\textstyle\sum_{i=0}^{15}i\,S_i\end{aligned}",
    "tal": r"t=s^{0{,}5}\,u^{6}",
    "bce": r"\mathcal{L}_{\text{cls}}=\mathrm{BCE}\bigl(\sigma(z),\,\hat t\bigr)",
    # --- §8 metrics
    "iou": r"\mathrm{IoU}=\dfrac{|B\cap B^{gt}|}{|B\cup B^{gt}|}\ \ge 0{,}5",
    "pr": r"P=\dfrac{TP}{TP+FP},\qquad R=\dfrac{TP}{TP+FN}",
    "ap": r"\mathrm{AP}=\dfrac{1}{101}\sum_{r\in\{0;\,0{,}01;\,\dots;\,1\}}\ \max_{r'\ge r}p(r')",
    "map": r"\mathrm{mAP}_{50\text{–}95}=\dfrac{1}{10}\sum_{\tau=0{,}50}^{0{,}95}\mathrm{AP}_{\tau}",
    "fp": r"\mathrm{FP}=\dfrac{1}{159}\sum_{f}\#\bigl\{b:\ s_b\ge\tau_{F1}\bigr\}",
    "boot": r"\mathrm{CI}_{95}=\bigl[q_{2{,}5};\ q_{97{,}5}\bigr]\ \text{của}\ \bigl\{\mathrm{mAP}(\mathcal{B}^{*}_{b})\bigr\}_{b=1}^{2000}",
    # --- §12 classic
    "cl_deblock": r"c=s\,w_e\,w_t\,\Delta,\quad \Delta=q_0-p_0",
    "cl_gate": r"g=\rho\bigl(\max(R,B);230;10\bigr)\,\rho\bigl(\min(R,B)-G;6;14\bigr)",
    "cl_rho": r"\rho(v;a,r)=\operatorname{clip}\bigl((v-a)/r;\,0;\,1\bigr)",
    "cl_depink": r"e=\max\Bigl(0,\dfrac{R+B}{2}-G\Bigr),\quad \delta=0{,}56\,\alpha\,e",
    "cl_depink2": r"R'=R-\delta,\quad B'=B-\delta,\quad G'=G+0{,}704\,\delta",
    "cl_vig": r"J^{c}=I^{c}\cdot\min\Bigl(\bigl(1+r^{2}/f^{2}\bigr)^{2},\ 1{,}35\Bigr)",
    "dc": r"I^{\mathrm{dark}}(x)=\min_{y\in\Omega_{15}(x)}\ \min_{c}\,I^{c}(y)",
    "air": r"A^{c}=\operatorname{mean}\bigl\{I^{c}(x):\ I^{\mathrm{dark}}(x)\ge P_{99{,}9}\bigr\}",
    "trans": r"\tilde t(x)=1-\omega\,\dfrac{I^{\mathrm{dark}}(x)}{\bar A},\quad \omega=0{,}85",
    "gf": r"\begin{aligned}q_i&=\bar a_k\,Y_i+\bar b_k\\ a_k&=\dfrac{\operatorname{cov}_k(Y,\tilde t)}{\operatorname{var}_k(Y)+\varepsilon},\quad b_k=\bar{\tilde t}_k-a_k\bar Y_k\end{aligned}",
    "gfp": r"r=6,\ \ \varepsilon=10^{-3},\ \ t=\operatorname{clip}(q;\,0{,}4;\,1)",
    "recov": r"J^{c}(x)=\dfrac{I^{c}(x)-A^{c}}{t(x)}+A^{c}",
    "he_book": r"\tilde t=1-\omega\min_{c}\min_{y\in\Omega}\dfrac{I^{c}(y)}{A^{c}},\qquad J=\dfrac{I-A}{\max(t,t_0)}+A",
    # --- §13 NAFNet
    "sg": r"\mathrm{SG}(X)=X_{1}\odot X_{2}",
    "sca": r"\mathrm{SCA}(X)=X\odot\bigl(W\,\mathrm{pool}(X)+b\bigr)",
    "nres": r"y=u+\beta\odot x,\qquad \mathrm{out}=y+\gamma\odot x'",
    "nloss": r"\mathcal{L}=\bigl\|\hat y-y\bigr\|_{1}+\bigl(1-\mathrm{SSIM}(\hat y,y)\bigr)",
    "gres": r"\hat y=x+\mathcal{N}_{\theta}(x)",
    # --- §14 image evaluation
    "pinkdrop": r"\Delta_{\text{hồng}}=1-\dfrac{\bar p_{\text{sau}}}{\bar p_{\text{gốc}}}\ \ge 0{,}9",
    "flick": r"F=\dfrac{\sum_i\overline{|P_{i+1}-P_i|}}{\sum_i\overline{|R_{i+1}-R_i|}}\ \le 1",
    "wilson": r"\dfrac{\hat p+\frac{z^{2}}{2n}\ \pm\ z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^{2}}{4n^{2}}}}{1+\frac{z^{2}}{n}},\quad z=1{,}96",
    "cl_alpha": r"\alpha=\mathrm{box}_{7\times7}\bigl(\mathrm{dilate}_{21\times21}(g)\bigr)",
    "stat1": r"\mathcal{L}=\dfrac{1}{|K|}\sum_{k\in K}\dfrac{W_1(\text{thật}_k,\ \text{giả}_k)}{\operatorname{IQR}(\text{thật}_k)}",
    "w1": r"W_1(a,b)=\int\bigl|F_a(x)-F_b(x)\bigr|\,dx",
    "pr2": r"\begin{aligned}P&=\dfrac{TP}{TP+FP}\\ R&=\dfrac{TP}{TP+FN}\end{aligned}",
    "ap2": r"\begin{aligned}\mathrm{AP}&=\dfrac{1}{101}\sum_{r}p_{\text{bao}}(r)\\ p_{\text{bao}}(r)&=\max_{r'\ge r}p(r')\end{aligned}",
    "boot2": r"\begin{aligned}\mathrm{CI}_{95}&=\bigl[q_{2{,}5};\ q_{97{,}5}\bigr]\\ &\text{của }\bigl\{\mathrm{mAP}(\mathcal{B}^{*}_{b})\bigr\}_{b=1}^{2000}\end{aligned}",
}

COLOR = {"obj_w": WHITE}
