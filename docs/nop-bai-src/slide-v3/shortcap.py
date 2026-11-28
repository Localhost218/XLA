import re,glob
B=chr(92)
tok=B+'caption{'
for p in sorted(glob.glob('sections/*.tex')):
    s=open(p,encoding='utf-8').read(); out=[]; i=0; n=0
    while True:
        j=s.find(tok,i)
        if j<0: out.append(s[i:]); break
        out.append(s[i:j]); k=j+len(tok); d=1; m=k
        while d:
            c=s[m]
            if c=='{': d+=1
            elif c=='}': d-=1
            m+=1
        body=s[k:m-1]
        # find ". " at depth 0
        d=0; cut=None
        for q,c in enumerate(body):
            if c=='{': d+=1
            elif c=='}': d-=1
            elif c=='.' and d==0 and body[q+1:q+2]==' ': cut=q; break
        # skip subfigure captions: check if inside subfigure (preceding text since last \begin{subfigure} not closed)
        pre=s[:j]
        insub=pre.rfind(B+'begin{subfigure}')>pre.rfind(B+'end{subfigure}')
        if cut and not insub and '$' not in body[:cut] or (cut and not insub and body[:cut].count('$')%2==0):
            out.append(B+'caption['+body[:cut]+']{'+body+'}'); n+=1
        else:
            out.append(s[j:m])
        i=m
    open(p,'w',encoding='utf-8',newline='\n').write(''.join(out)); print(p,n)
