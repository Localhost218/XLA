B=chr(92)
def rep(p,pairs):
    s=open(p,encoding='utf-8').read()
    for a,b in pairs:
        a=a.replace('@',B); b=b.replace('@',B)
        assert a in s,(p,a)
        s=s.replace(a,b)
    open(p,'w',encoding='utf-8',newline='\n').write(s)
rep('main.tex',[('    tick label style={font=@scriptsize},','    tick label style={font=@scriptsize}, scaled ticks=false, /pgf/number format/fixed,')])
rep('fig/pgf-realism.tex',[('horizontal sep=5mm, yticklabels at=edge left},\n  realbar, symbolic y coords={tt','horizontal sep=7mm, yticklabels at=edge left},\n  realbar, symbolic y coords={tt'),
 ('{@scriptsize Khoảng cách thống kê (nhỏ = giống)}','{@scriptsize Khoảng cách (nhỏ = giống)}'),
 ('{@scriptsize C2ST (0,5 = không tách được)}','{@scriptsize C2ST (0,5 = lý tưởng)}'),
 ('width=4.9cm, height=3.7cm','width=4.6cm, height=3.7cm')])
rep('fig/tikz-e2.tex',[('{@textbf{E1 D2} (YOLO11s)@@dev-val 0,329}','{@textbf{E1 D2} (YOLO11s)@@dev-val 0,329@@làm thầy vòng 1}'),
 ('@draw[mui] (e1) -- node[above, font=@tiny] {thầy vòng 1} (nhan1);','@draw[mui] (e1) -- (nhan1);'),
 ('{so chất lượng nhãn giả, thua}','{nhãn giả kém hơn}')])
rep('fig/tikz-classic.tex',[('text width=17mm, minimum height=14mm}','text width=17mm, minimum height=17mm}'),
 ('@fill[@mau!80] ($(n@i.south west)+(0.05,-0.22)$) rectangle ++(@ms*0.6,-0.22);','@fill[@mau!80] ($(n@i.west |- 0,-0.98)+(0.05,0)$) rectangle ++(@ms*0.6,-0.2);'),
 ('at ($(n@i.south west)+(0.05+@ms*0.6,-0.33)$)','at ($(n@i.west |- 0,-1.08)+(0.05+@ms*0.6,0)$)'),
 ('at ($(n0.south west)+(0,-0.55)$)','at ($(n0.west |- 0,-1.3)$)')])
s=open('fig/pgf-scatter.tex',encoding='utf-8').read()
pairs=[('@node[font=@scriptsize, cLuc!70!black, anchor=north west, align=left] at (axis cs:0.06,0.338) {vùng đạt tiêu chí:@@mAP $@geq$ 0,227 và@@báo nhầm $@leq$ 0,05};',
        '@node[font=@scriptsize, cLuc!70!black, anchor=north east, align=right] at (axis cs:2.12,0.338) {dải lục sát trục tung: vùng đạt tiêu chí@@(mAP $@geq$ 0,227 và báo nhầm $@leq$ 0,05)};'),
 ('@node[font=@scriptsize, anchor=north west, font=@scriptsize@bfseries] at (axis cs:0.31,0.231) {E1 D2};','@node[font=@scriptsize@bfseries, anchor=east] at (axis cs:0.27,0.236) {E1 D2};'),
 ('@node[font=@scriptsize, anchor=north east] at (axis cs:0.38,0.214) {thầy 11m};','@node[font=@scriptsize, anchor=west] at (axis cs:0.43,0.214) {thầy 11m};')]
open('fig/pgf-scatter.tex','w',encoding='utf-8',newline='\n').write(s)
rep('fig/pgf-scatter.tex',pairs)
print('ok')
