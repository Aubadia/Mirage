"""Acquire the pinned standalone TeX tool inside the report directory only."""
import hashlib,json,pathlib,tarfile,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
target=ROOT/'cache/tectonic.tar.gz'
if not target.is_file():urllib.request.urlretrieve(url,target)
with tarfile.open(target) as t:
 for m in t.getmembers():
  if m.name=='tectonic':t.extract(m,ROOT/'tools',filter='data')
p=ROOT/'tools/tectonic'
(ROOT/'data/tex-tool.json').write_text(json.dumps({'version':'0.17.0','url':url,'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
