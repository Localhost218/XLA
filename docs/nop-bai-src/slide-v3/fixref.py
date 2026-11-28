p='sections/09-tai-lieu.tex'
lines=open(p,encoding='utf-8').read().split('\n')
B=chr(92)
lines[0]=B+'renewcommand{'+B+'refname}{Tài liệu tham khảo}'
open(p,'w',encoding='utf-8',newline='\n').write('\n'.join(lines))
