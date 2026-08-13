# Retained reference-format artifact

- Design authority: `Manuscript for review.docx` supplied by the user.
- Reference SHA-256: `F6341D1B87AC2413399351C82574C1A5BD526E268440E15C051B6DC84121ACD9`.
- Reference rendering: Microsoft Word PDF export, 31 A4 portrait pages, one section.
- Output target: the project LaTeX manuscript and its compiled PDF; the reference DOCX remains read-only.

## Measured design rules

- Page: A4 portrait, one column.
- Margins: 1 in top/bottom and 1.25 in left/right.
- Typeface: Times New Roman family; 12 pt body, approximately 15.6 pt leading.
- Title: 16 pt bold and centered.
- Author block: centered authors, left-aligned affiliation and corresponding-author line.
- Abstract and keywords: inline bold labels followed by justified body text.
- Headings: 14 pt bold for levels 1 and 2; 12 pt bold for level 3.
- Paragraphs: justified with a first-line indent and no extra paragraph spacing.
- End matter: Author Contributions, Acknowledgements, Conflict of Interest Disclosure, References, caption inventories, tables, then figures.
- Tables: captions above; restrained black-and-white three-line style.
- Figures: captions below; color is retained where it communicates scientific series or model branches.
- Page numbers: centered footer.

## Content-slot mapping

- Title -> existing GHXTox manuscript title.
- Author names -> Liuyu Xu, Kehan Gao, Lechen Hao, and Shengli Zhang, in that order.
- Affiliation -> School of Mathematics and Statistics, Xidian University, Xi'an 710071, P. R. China, matching the supplied sample.
- Corresponding author -> Shengli Zhang; email -> shengli0201@163.com, matching the supplied sample.
- Abstract, keywords, body, equations, references, tables, and figures -> existing verified project content.
- Author contributions -> `All authors contributed equally.`; acknowledgement and grant numbers match the supplied sample.
- Conflict disclosure -> explicit no-conflict statement.

## Verification requirements

- Compile without undefined references or box-overflow warnings.
- Render every output page and inspect the title page, references, caption lists, tables, and figures.
- Do not modify the supplied reference DOCX.
