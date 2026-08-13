# GHXTox manuscript format audit against the supplied DOCX

## Scope

- Reference: `Manuscript for review.docx` (read-only), rendered with Microsoft Word.
- Project manuscript: `v1.pdf` generated from `v1.tex`.
- This is a formatting comparison against the supplied sample, not a claim about a journal's official author guidelines.

## Reference format distilled

- A4 portrait, one column.
- Margins: top/bottom 72 pt (25.4 mm), left/right 90 pt (31.75 mm).
- Main title: Times New Roman, 16 pt, bold, centered.
- Author and affiliation block appears below the title; corresponding-author line is explicit.
- Abstract begins with an inline bold `Abstract.` label; the text is 12 pt Times New Roman and fully justified.
- Keywords begin with an inline bold `Keywords:` label.
- Level-1 headings: 14 pt bold, flush left, uppercase, numbered.
- Level-2 headings: 14 pt bold, flush left, numbered.
- Level-3 headings: 12 pt bold, flush left, numbered.
- Body: Times New Roman, 12 pt, justified, first-line indent, approximately 15.6 pt line spacing.
- Main text contains citations and figure/table callouts, but the actual tables and figures are collected after the reference list. Separate `Table Captions` and `Figure Captions` lists precede those end matter items.
- Back matter includes Author Contributions, Acknowledgements, Conflict of Interest Disclosure, and References.

## Findings

### Critical before template-matched submission

1. **Missing author metadata.** The project title page has no authors, affiliations, corresponding-author marker, or email. The supplied sample requires all four items.
2. **Main document flow differs.** The project inserts figures and tables in the body; the sample places table captions/tables and figure captions/figures after the references. If the sample is binding, the project must be reorganized.
3. **Typography does not match.** The project uses Latin Modern at 11 pt; the sample uses Times New Roman at 12 pt for body text, 14 pt for first/second-level headings, 12 pt for third-level headings, and 16 pt bold for the title.
4. **Page geometry does not match.** Both are A4 and single-column, but the project uses 25 mm margins on all sides, whereas the sample uses 25.4 mm top/bottom and 31.75 mm left/right. The project text block is therefore about 13.5 mm wider.
5. **Abstract format does not match.** The project uses a centered standalone `Abstract` heading and a smaller/narrower abstract block. The sample uses an inline bold `Abstract.` label followed by full-width justified 12 pt text.

### Major

6. **Heading hierarchy differs.** Project headings are larger LaTeX article defaults; the sample uses restrained 14/14/12 pt numbered headings. Project `Data and Software Availability` and `Declaration of Competing Interest` are unnumbered in a larger heading style, unlike the sample's compact back-matter headings.
7. **Back matter is incomplete.** The project contains Data and Software Availability and Declaration of Competing Interest, but lacks Author Contributions and Acknowledgements. The sample includes both.
8. **Figure/table captions differ.** The project uses LaTeX float captions with colored table highlights. The sample uses plain black three-line tables, captions above tables, captions below figures, and separate caption inventories in the end matter. Colored fills are not present in the sample.
9. **Body paragraph rhythm differs.** The project is denser, with smaller type and tighter apparent leading. The sample uses 12 pt Times New Roman, justified text, first-line indentation, and approximately 15.6 pt line spacing.
10. **References differ in style.** Both are numbered by appearance, but the project's `unsrt` output uses full given names and italicized full journal titles, while the sample uses surname/initial formatting, semicolon-separated authors, abbreviated journal titles, and a more compact entry style.

### Minor or conditional

11. The project already matches A4 portrait, single-column layout, numbered main sections/subsections, bracketed numerical citations, equation numbering, table captions above tables, and figure captions below figures.
12. The project page numbers are centered in the footer, consistent with the sample's body pagination.
13. The project's figures and tables are visually clean and readable, but their blue-orange styling is a deliberate departure from the sample's restrained black-and-white tabular style.
14. The sample itself appears to contain one blank/end-transition page during Word PDF export; that artifact should not be copied as a design requirement.

## Recommended correction order

1. Add complete authorship, affiliation, and corresponding-author metadata.
2. Change the page and font system to the sample's A4/TNR/12 pt/31.75 mm side-margin specification.
3. Rebuild the title, abstract, keyword, and heading styles.
4. Add Author Contributions and Acknowledgements.
5. Decide whether the sample's end-matter placement of all tables and figures is mandatory; if yes, move floats and add separate caption lists.
6. Replace `unsrt` with a bibliography style that reproduces the sample's author initials, punctuation, and abbreviated journals.
7. Remove colored table fills if strict visual fidelity to the sample is required; keep color in figures only if allowed by the destination journal.
