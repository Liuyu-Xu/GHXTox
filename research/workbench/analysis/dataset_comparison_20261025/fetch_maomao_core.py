"""Select metadata CSVs from the official ZIP using HTTP byte ranges.
Preserve source bytes and ZIP CRC; no need to download embeddings or 71k cards.
"""
from pathlib import Path
import io,json,requests,zipfile,hashlib,time
OUT=Path(__file__).resolve().parent/'sources/kren-ai-lab__maomao'

class RemoteFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.session=requests.Session();self.log=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b''
        start=self.pos;end=start+n-1
        r=self.session.get(self.url,params={'audit_range':f'{start}-{end}'},headers={'Range':f'bytes={start}-{end}'},timeout=60)
        assert r.status_code==206,(r.status_code,r.text[:100])
        assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),r.headers
        b=r.content;assert len(b)==n;self.pos+=n
        self.log.append({'start':start,'end':end,'sha256':hashlib.sha256(b).hexdigest()})
        return b

def run():
    meta=json.loads((OUT/'zenodo_record.json').read_text(encoding='utf8'))
    entry=next(x for x in meta['files'] if x['key']=='core_layer.zip')
    f=RemoteFile(entry['links']['self'],entry['size']);z=zipfile.ZipFile(f)
    targets=['maomao_sequence_pivot.csv','maomao_toxicity_measurements.csv','metadata_toxicity_properties.json','audit_endpoint_counts.csv']
    result=[]
    for name in targets:
        found=[x for x in z.infolist() if x.filename.endswith('/'+name)]
        assert len(found)==1,(name,len(found));x=found[0]
        b=z.read(x);path=OUT/name;path.write_bytes(b)
        result.append(dict(zip_path=x.filename,local=str(path),bytes=len(b),crc32=f'{x.CRC:08x}',sha256=hashlib.sha256(b).hexdigest()))
        print(name,len(b),flush=True)
    (OUT/'selective_download_manifest.json').write_text(json.dumps(dict(url=entry['links']['self'],archive_bytes=entry['size'],
        archive_md5_expected=entry['checksum'],archive_full_md5_verified=False,selected_file_crc_verified=True,
        range_requests=f.log,selected_files=result,transfer_bytes=sum(x['end']-x['start']+1 for x in f.log)),indent=2),encoding='utf8')

if __name__=='__main__':run()
