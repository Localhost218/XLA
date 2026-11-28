n=[86,235,50,12,287,267,50,10,123,108,75,33,138,77,63,72,248,147,210,174,124,97,165,64]
b=['>14','>14','3-8','>14','8-14','8-14','3-8','8-14','>14','3-8','>14','>14','8-14','3-8','8-14','3-8','>14','3-8','>14','>14','8-14','8-14','8-14','3-8']
test={0,1,3,4,6,7,9,10,14,15,16,20,23}
W=15.0; s=W/sum(n); x=0; out=[]
binc={'3-8':'cVang!55','8-14':'cVang!85!cCam','>14':'cCam!70!cHong'}
BS=chr(92)
for i,(k,bb) in enumerate(zip(n,b)):
    w=max(k*s,0.10)
    col='cCam' if i in test else 'cXanh'
    out.append(f"{BS}fill[{col}] ({x:.3f},0) rectangle ++({w-0.03:.3f},0.9);")
    out.append(f"{BS}fill[{binc[bb]}] ({x:.3f},-0.32) rectangle ++({w-0.03:.3f},0.22);")
    if w>0.35: out.append(f"{BS}node[font={BS}tiny,white] at ({x+w/2-0.015:.3f},0.45) {{{i}}};")
    x+=w
out.append(f"{BS}def{BS}tlW{{{x:.3f}}}")
open(r'D:\XLA\docs\nop-bai-src\latex\fig\timeline-blocks.tex','w',newline='\n').write("\n".join(out)+"\n")
