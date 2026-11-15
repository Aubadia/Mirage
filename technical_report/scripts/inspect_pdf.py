"""Inspect report PDF metadata/text and render every page for review, under Slurm."""
import os,sys,pathlib,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Run rendering in Slurm.')
sys.path.insert(0,str(ROOT/'tools/python'))
from pypdf import PdfReader
import pypdfium2 as pdfium
reader=PdfReader(ROOT/'main.pdf');doc=pdfium.PdfDocument(str(ROOT/'main.pdf'))
text=[];pages=[];fonts={}
def embedded(font):
 f=font.get_object()
 if '/DescendantFonts' in f:return all(embedded(x) for x in f['/DescendantFonts'])
 desc=f.get('/FontDescriptor')
 return bool(desc and any(k in desc.get_object() for k in ['/FontFile','/FontFile2','/FontFile3']))
for i,p in enumerate(reader.pages):
 t=p.extract_text();text.append(t)
 size=[float(x) for x in p.mediabox];pages.append({'page':i+1,'mediabox':size,'text_characters':len(t)})
 image=doc[i].render(scale=1.4).to_pil();image.save(ROOT/'validation'/f'page-{i+1:02d}.png')
 for name,font in p.get('/Resources',{}).get('/Font',{}).items():
  f=font.get_object();fonts[str(f.get('/BaseFont',name))]={'subtype':str(f.get('/Subtype')),'embedded':embedded(f)}
alltext='\n\n'.join(text);(ROOT/'validation/pdf-text.txt').write_text(alltext)
assert len(reader.pages)>=5
assert all(f['embedded'] for f in fonts.values()),fonts
assert 'Roadmap' in alltext and '008' in alltext and '009' in alltext and '010' in alltext
assert all(abs(x['mediabox'][2]-612)<1 and abs(x['mediabox'][3]-792)<1 for x in pages)
assert not any(word in alltext for word in ['CITATION NEEDED','PLACEHOLDER','Lorem ipsum'])
result={'job_id':os.environ['SLURM_JOB_ID'],'pages':pages,'fonts':fonts,'paper_size':'US Letter','sha256':hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest(),'rendered_every_page':True,'scope':'Static PDF checks; not an IEEE PDF eXpress certificate.'}
(ROOT/'validation/pdf-inspection.json').write_text(json.dumps(result,indent=2)+'\n');print('PDF:',len(pages),'pages; all rendered; letter size checked.')
