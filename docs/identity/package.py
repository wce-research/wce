"""Package exact browser-rendered favicons and a deterministic identity archive."""
from pathlib import Path
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent

def build():
    for folder in sorted((ROOT/'finalists').iterdir()):
        bitmaps=[(size,(folder/f'favicon-{size}.png').read_bytes()) for size in (16,32,48)]
        offset=6+16*len(bitmaps); directory=[]
        for size,data in bitmaps:
            directory.append(struct.pack('<BBBBHHII',size,size,0,0,1,32,len(data),offset))
            offset+=len(data)
        (folder/'favicon.ico').write_bytes(struct.pack('<HHH',0,1,len(bitmaps))+b''.join(directory)+b''.join(data for _,data in bitmaps))
    files=[]
    selected=[ROOT/'index.html',ROOT/'README.md',ROOT/'concepts.json',ROOT/'assets/typeface-license.txt']
    selected+=sorted((ROOT/'concepts').glob('*.svg'))
    selected+=sorted((ROOT/'finalists').rglob('*.*'))
    selected+=[ROOT/'boards'/f'{name}.{ext}' for name in ('initial-12','finalists-5') for ext in ('svg','png')]
    for file in selected:
        data=file.read_bytes();entry={'file':str(file.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        if file.suffix=='.png':
            assert data[:8]==b'\x89PNG\r\n\x1a\n',file
            entry['width'],entry['height']=struct.unpack('>II',data[16:24])
        if file.suffix=='.svg':
            svg=ET.fromstring(data);entry['width']=int(svg.attrib['width']);entry['height']=int(svg.attrib['height'])
            assert not any(node.tag.endswith('script') for node in svg.iter()),file
        if file.name.startswith('github-preview') and file.suffix=='.png':assert len(data)<1_000_000
        if file.name.startswith('linkedin-') and file.suffix=='.png':assert len(data)<3_000_000
        files.append(entry)
    (ROOT/'manifest.json').write_text(json.dumps({'study':'WCE identity 01','finalists':['a2','b2','c2','d2','f2'],'files':files},ensure_ascii=False,indent=2))
    selected.append(ROOT/'manifest.json')
    with zipfile.ZipFile(ROOT/'wce-identity-study-01.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for file in selected:
            info=zipfile.ZipInfo('wce-identity-study-01/'+str(file.relative_to(ROOT)),date_time=(2026,10,3,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,file.read_bytes())
    print(f'Validated {len(files)} assets; packaged five ICOs and full study ZIP.')

if __name__=='__main__':build()
