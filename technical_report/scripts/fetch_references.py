"""Fetch bibliographic records; never synthesize scholarly metadata from memory."""
import json,re,urllib.request,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
records=[('nablafuzz','10.1109/ICSE48619.2023.00105','https://arxiv.org/abs/2302.04351'),('gpufuzz','10.48550/arXiv.2602.10478','https://arxiv.org/abs/2602.10478')]
notes=[]; bib=[]
for key,doi,primary in records:
 endpoints=[('https://doi.org/'+doi,{'Accept':'application/x-bibtex'}),('https://api.crossref.org/works/'+doi+'/transform/application/x-bibtex',{}),('https://api.datacite.org/dois/'+doi+'/media/application/x-bibtex',{})]
 errors=[]
 for url,headers in endpoints:
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'MirageTechnicalReport/1.0',**headers})
   with urllib.request.urlopen(req,timeout=40) as r: text=r.read().decode()
   if not re.match(r'\s*@',text): raise ValueError('Not BibTeX')
   (ROOT/'data'/f'{key}.original.bib').write_text(text)
   normalized=re.sub(r'(@\w+\s*\{)[^,]+,',lambda m:m[1]+key+',',text,count=1)
   bib.append(normalized)
   notes.append({'key':key,'doi':doi,'retrieved_from':url,'primary_verification':primary,'status':'fetched','prior_errors':errors})
   break
  except Exception as e:errors.append(str(e))
 else:raise RuntimeError((key,errors))
(ROOT/'references.bib').write_text('\n\n'.join(bib)+'\n')
(ROOT/'data/citation-verification.json').write_text(json.dumps(notes,indent=2)+'\n')
print([(n['key'],n['status']) for n in notes])
