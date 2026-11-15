"""Unpack report-only PDF inspection wheels; does not change any environment."""
import hashlib,json,pathlib,urllib.request,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
notes=[]
for package,version in [('pypdf','7.1.0'),('pypdfium2','5.15.2')]:
 with urllib.request.urlopen('https://pypi.org/pypi/'+package+'/'+version+'/json',timeout=40) as r:info=json.load(r)
 candidates=[x for x in info['urls'] if x['filename'].endswith('.whl') and ('py3-none-any' in x['filename'] or 'py3-none-manylinux' in x['filename'] and 'x86_64' in x['filename'])]
 if not candidates:raise RuntimeError(package)
 a=candidates[0];target=ROOT/'cache'/a['filename']
 urllib.request.urlretrieve(a['url'],target)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==a['digests']['sha256']
 with zipfile.ZipFile(target) as z:z.extractall(ROOT/'tools/python')
 notes.append({'package':package,'version':info['info']['version'],'filename':a['filename'],'url':a['url'],'sha256':a['digests']['sha256']})
(ROOT/'data/pdf-tools.json').write_text(json.dumps(notes,indent=2)+'\n')
print([(x['package'],x['version']) for x in notes])
