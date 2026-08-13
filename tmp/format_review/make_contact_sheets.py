from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def build(source: Path, out_dir: Path, prefix: str) -> None:
    files = sorted(source.glob("page-*.png"))
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_w = 248
    label_h = 28
    gap = 12
    cols, rows = 4, 2
    per_sheet = cols * rows
    font = ImageFont.load_default()
    for start in range(0, len(files), per_sheet):
        batch = files[start:start + per_sheet]
        thumbs = []
        max_h = 0
        for path in batch:
            image = Image.open(path).convert("RGB")
            thumb_h = round(image.height * thumb_w / image.width)
            image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            thumbs.append((path, image))
            max_h = max(max_h, thumb_h)
        sheet = Image.new(
            "RGB",
            (cols * thumb_w + (cols + 1) * gap,
             rows * (max_h + label_h) + (rows + 1) * gap),
            "#ECEFF3",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (path, image) in enumerate(thumbs):
            row, col = divmod(index, cols)
            x = gap + col * (thumb_w + gap)
            y = gap + row * (max_h + label_h + gap)
            sheet.paste(image, (x, y))
            label = path.stem.replace("page-", "Page ")
            draw.text((x + 4, y + max_h + 7), label, fill="#222222", font=font)
        sheet.save(out_dir / f"{prefix}_{start // per_sheet + 1:02d}.png")


build(Path("tmp/format_review/template/render"), Path("tmp/format_review/contact"), "template")
build(Path("tmp/format_review/project/render"), Path("tmp/format_review/contact"), "project")
build(Path("tmp/format_review/final/render"), Path("tmp/format_review/contact_final"), "final")
build(Path("tmp/format_review/final2/render"), Path("tmp/format_review/contact_final2"), "final")
