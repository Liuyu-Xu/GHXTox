import pathlib, urllib.request, json, hashlib, zipfile, concurrent.futures, time
ROOT = pathlib.Path(__file__).resolve().parent

def fetch(model):
    url=f'https://webs.iiitd.edu.in/raghava/{model}/download/Model.zip'
    target=ROOT/model/'Model.zip'
    target.parent.mkdir(exist_ok=True)
    start=time.time()
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as res:
        expected=int(res.headers['Content-Length'])
    if not target.exists() or target.stat().st_size!=expected:
        parts=target.parent/'download_parts';parts.mkdir(exist_ok=True)
        chunk_size=8*1024*1024
        def part(a):
            b=min(expected-1,a+chunk_size-1);f=parts/f'{a}.part'
            if not f.exists() or f.stat().st_size!=b-a+1:
                for attempt in range(3):
                    req=urllib.request.Request(url+f'?audit_range={a}_{b}',headers={'Range':f'bytes={a}-{b}'})
                    try:
                        with urllib.request.urlopen(req,timeout=60) as r:
                            data=r.read()
                            assert r.status==206 and len(data)==b-a+1,(r.status,len(data))
                        f.write_bytes(data);break
                    except Exception:
                        if attempt==2:raise
            return f
        with concurrent.futures.ThreadPoolExecutor(5) as pool:
            pieces=list(pool.map(part,range(0,expected,chunk_size)))
        tmp=target.with_suffix('.part')
        with tmp.open('wb') as dst:
            for f in pieces:
                with f.open('rb') as src:
                    while chunk:=src.read(1024*1024):dst.write(chunk)
        assert tmp.stat().st_size==expected
        tmp.replace(target)
    members=[]
    with zipfile.ZipFile(target) as z:
        for i in z.infolist():
            if i.is_dir():continue
            path=(target.parent/i.filename).resolve()
            if not path.is_relative_to(target.parent.resolve()):raise ValueError(i.filename)
            members.append({'name':i.filename,'bytes':i.file_size})
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                with z.open(i) as src,path.open('wb') as dst:
                    while chunk:=src.read(4*1024*1024):dst.write(chunk)
    result={'url':url,'zip_bytes':target.stat().st_size,'zip_sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(target.read_bytes()).hexdigest(),'seconds':time.time()-start,'members':members}
    (ROOT/(model+'_asset_manifest.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(model,'extracted',members,flush=True)
    return result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(fetch,['hemopi2','ntxpred2']))
