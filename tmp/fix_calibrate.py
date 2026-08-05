p = "scripts/calibrate_lora_mamba.py"
s = open(p, encoding="utf-8").read()
old = '''    oofs = []
    for path in args.oof_jsons:
        payload = json.load(open(path, encoding="utf-8"))
        oofs.append(np.asarray(payload["oof"], dtype=np.float64))
    p_oof = np.mean(oofs, axis=0)
    labels = np.asarray(oofs[0] and json.load(open(args.oof_jsons[0], encoding="utf-8"))["labels"], dtype=int)'''
new = '''    oofs = []
    labels = None
    for path in args.oof_jsons:
        payload = json.load(open(path, encoding="utf-8"))
        oofs.append(np.asarray(payload["oof"], dtype=np.float64))
        if labels is None:
            labels = np.asarray(payload["labels"], dtype=int)
    p_oof = np.mean(oofs, axis=0)'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("fixed labels loading")
