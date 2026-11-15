"""Export the report's prose, equations, tables and figures to English Markdown."""
import pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1]
s=(ROOT/'main.tex').read_text()
for macro in ['Pre','In','Out','Obs']:s=s.replace('\\'+macro, '\\operatorname{'+macro+'}')
def arg(text,command):
 start=text.index(command)+len(command)
 while text[start].isspace():start+=1
 assert text[start]=='{'
 depth=1;i=start+1
 while depth:
  if text[i]=='{' and text[i-1]!='\\':depth+=1
  elif text[i]=='}' and text[i-1]!='\\':depth-=1
  i+=1
 return text[start+1:i-1],i
labels={};counters={}
for match in re.finditer(r'\\label\{([^}]+)\}',s):
 label=match[1];kind=label.split(':')[0];counters[kind]=counters.get(kind,0)+1
 labels[label]=str(counters[kind])
for match in re.finditer(r'label=\{(lst:[^}]+)\}',s):labels[match[1]]='1'
def inline(t):
 for cmd,fmt in [('texttt','`{}`'),('emph','*{}*'),('textbf','**{}**')]:
  t=re.sub(r'\\'+cmd+r'\{([^{}]*)\}',lambda m:fmt.format(m[1]),t)
 t=re.sub(r'\\(?:ref|eqref)\{([^}]+)\}',lambda m:labels.get(m[1],m[1]),t)
 t=re.sub(r'~?\\cite\{([^}]+)\}',lambda m:' '+', '.join('['+k+']' for k in m[1].split(',')),t)
 return t.replace('\\Mirage','Mirage').replace('\\%','%').replace('\\_','_').replace('\\&','&').replace('\\emph','').replace('---','—').replace('--','–').replace('~',' ')
protected={}
def protect(value):
 k=f'@@BLOCK{len(protected)}@@';protected[k]=value;return k
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',s,re.S)[1]
s=s.split('\\end{IEEEkeywords}',1)[1].split('\\bibliographystyle',1)[0]
# Code listings and display mathematics are preserved verbatim.
s=re.sub(r'\\begin\{lstlisting\}(?:\[.*?\])?(.*?)\\end\{lstlisting\}',lambda m:protect('\n```python\n'+m[1].strip()+'\n```\n'),s,flags=re.S)
s=re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}',lambda m:protect('\n$$\n'+re.sub(r'\\label\{[^}]+\}','',m[1]).strip()+'\n$$\n'),s,flags=re.S)
def table(m):
 block=m[0];caption,_=arg(block,'\\caption');start=block.index('\\begin{tabular}')+len('\\begin{tabular}')
 _,offset=arg(block[start:],'')
 body=block[start+offset:block.index('\\end{tabular}')]
 body=re.sub(r'\\(?:toprule|midrule|bottomrule)','',body)
 rows=[]
 for line in re.split(r'\\\\',body):
  line=line.strip()
  if not line:continue
  rows.append([inline(c.strip()).replace('\n',' ') for c in line.split('&')])
 width=max(len(r) for r in rows)
 rows=[r+['']*(width-len(r)) for r in rows]
 lines=['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*width)+' |']
 lines+=['| '+' | '.join(r)+' |' for r in rows[1:]]
 return protect('\n**'+inline(caption)+'**\n\n'+'\n'.join(lines)+'\n')
s=re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',table,s,flags=re.S)
def figure(m):
 b=m[0];caption,_=arg(b,'\\caption');filename=re.search(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}',b)[1].replace('.pdf','.png')
 return protect('\n![Quantitative comparison]('+filename+')\n\n'+inline(caption)+'\n')
s=re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',figure,s,flags=re.S)
s=inline(s)
for cmd,prefix in [('subsection','### '),('section','## ')]:s=re.sub(r'\\'+cmd+r'\{([^{}]*)\}',lambda m:'\n'+prefix+m[1]+'\n',s)
s=re.sub(r'\\paragraph\{([^{}]*)\}',lambda m:'\n**'+m[1]+'** ',s)
s=s.replace('\\begin{itemize}','').replace('\\end{itemize}','').replace('\\item','\n- ')
s=re.sub(r'\\label\{[^}]*\}','',s).replace('\\appendices','\n## Appendix\n')
s=re.sub(r'\n{3,}','\n\n',s)
s=re.sub(r'(?<!\n)\n(?!\n|[#|\-!`@$])',' ',s)
for k,v in protected.items():s=s.replace(k,v)
refs='''
## References and local evidence

- [rm008] [Roadmap 008](../docs/roadmap-008-joint-state-exploration.md).
- [rm009] [Roadmap 009](../docs/roadmap-009-general-state-research.md).
- [rm010] [Roadmap 010](../docs/roadmap-010-optimizer-gradient-chain.md).
- [result008] [Frozen 008 comparison](data/phase-008-comparison.json).
- [result009] [Frozen 009 comparison](data/phase-009-comparison.json).
- [result010] [Frozen 010 comparison](data/phase-010-comparison.json).
- [addcase] [Saved add evidence](data/example-add-evidence.json) and [exact source](data/example-add-program.py).
- [nablafuzz] C. Yang et al., *Fuzzing Automatic Differentiation in Deep-Learning Libraries*, ICSE 2023. [DOI](https://doi.org/10.1109/ICSE48619.2023.00105).
- [gpufuzz] Z. Li et al., *GPU-Fuzz: Finding Memory Errors in Deep Learning Frameworks*, 2026. [arXiv](https://arxiv.org/abs/2602.10478).

Full fetched metadata appears in [references.bib](references.bib). [Evidence index](data/evidence-index.md) maps repository artifacts and hashes. The LaTeX/PDF is the typeset authority; this Markdown is generated from its content.
'''
(ROOT/'report.md').write_text('# How Mirage Tests One API: Stateful Exploration, Runtime Evidence, and Conditional Oracles\n\nEnglish technical report, November 15, 2026.\n\n## Abstract\n\n'+inline(abstract).strip()+'\n\n'+s.strip()+'\n'+refs)
print('Exported English report.md')
