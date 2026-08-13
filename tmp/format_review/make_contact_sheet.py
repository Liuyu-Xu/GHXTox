from pathlib import Path

from PIL import Image, ImageDraw


source = Path("tmp/format_review/final_authors/render")
target = Path("tmp/format_review/final_authors/contact")
target.mkdir(parents=True, exist_ok=True)
pages = sorted(source.glob("page-*.png"))

thumb_width = 300
label_height = 30
gap = 14
cols = 4
rows = 2

for group_index in range(0, len(pages), cols * rows):
    group = pages[group_index : group_index + cols * rows]
    thumbs = []
    for page in group:
        image = Image.open(page).convert("RGB")
        height = round(image.height * thumb_width / image.width)
        image.thumbnail((thumb_width, height), Image.Resampling.LANCZOS)
        thumbs.append((page, image))

    cell_height = max(image.height for _, image in thumbs) + label_height
    sheet = Image.new(
        "RGB",
        (cols * thumb_width + (cols + 1) * gap, rows * cell_height + (rows + 1) * gap),
        "#e8edf2",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (page, image) in enumerate(thumbs):
        row, col = divmod(index, cols)
        x = gap + col * (thumb_width + gap)
        y = gap + row * (cell_height + gap)
        sheet.paste(image, (x, y))
        draw.text((x, y + image.height + 5), page.stem.replace("page-", "Page "), fill="black")

    sheet.save(target / f"contact_{group_index // (cols * rows) + 1:02d}.png")
