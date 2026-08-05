import zipfile, re, sys, os

def extract_docx(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    lines = []
    for p in paras:
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)
        line = "".join(texts)
        if line.strip():
            lines.append(line)
    return lines

for path in sys.argv[1:]:
    lines = extract_docx(path)
    out = os.path.join("tmp", "docx_" + os.path.splitext(os.path.basename(path))[0] + ".txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", out, len(lines), "lines")
