param(
    [Parameter(Mandatory = $true)][string]$InputDocx,
    [Parameter(Mandatory = $true)][string]$OutputPdf,
    [Parameter(Mandatory = $true)][string]$OutputJson
)

$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($InputDocx, $false, $true)

    $sections = @()
    foreach ($section in $document.Sections) {
        $setup = $section.PageSetup
        $sections += [ordered]@{
            index = $section.Index
            page_width_pt = [math]::Round($setup.PageWidth, 2)
            page_height_pt = [math]::Round($setup.PageHeight, 2)
            top_margin_pt = [math]::Round($setup.TopMargin, 2)
            bottom_margin_pt = [math]::Round($setup.BottomMargin, 2)
            left_margin_pt = [math]::Round($setup.LeftMargin, 2)
            right_margin_pt = [math]::Round($setup.RightMargin, 2)
            header_distance_pt = [math]::Round($setup.HeaderDistance, 2)
            footer_distance_pt = [math]::Round($setup.FooterDistance, 2)
            orientation = [int]$setup.Orientation
            columns = $setup.TextColumns.Count
        }
    }

    $paragraphs = @()
    $limit = [math]::Min($document.Paragraphs.Count, 5000)
    for ($i = 1; $i -le $limit; $i++) {
        $p = $document.Paragraphs.Item($i)
        $text = ($p.Range.Text -replace '[\r\a]+$', '').Trim()
        if (-not $text) { continue }
        $paragraphs += [ordered]@{
            index = $i
            page = [int]$p.Range.Information(3)
            text = if ($text.Length -gt 240) { $text.Substring(0, 240) } else { $text }
            style = [string]$p.Range.Style.NameLocal
            font = [string]$p.Range.Font.Name
            size_pt = [double]$p.Range.Font.Size
            bold = [int]$p.Range.Font.Bold
            italic = [int]$p.Range.Font.Italic
            alignment = [int]$p.Alignment
            line_spacing_rule = [int]$p.LineSpacingRule
            line_spacing_pt = [math]::Round($p.LineSpacing, 2)
            space_before_pt = [math]::Round($p.SpaceBefore, 2)
            space_after_pt = [math]::Round($p.SpaceAfter, 2)
            first_line_indent_pt = [math]::Round($p.FirstLineIndent, 2)
        }
    }

    $tables = @()
    foreach ($table in $document.Tables) {
        $tables += [ordered]@{
            index = $table.Index
            page = [int]$table.Range.Information(3)
            rows = $table.Rows.Count
            columns = $table.Columns.Count
            preferred_width_type = [int]$table.PreferredWidthType
            preferred_width = [math]::Round($table.PreferredWidth, 2)
            alignment = [int]$table.Rows.Alignment
            autofit_behavior = [int]$table.AllowAutoFit
        }
    }

    $shapes = @()
    foreach ($shape in $document.InlineShapes) {
        $shapes += [ordered]@{
            index = $shape.Index
            page = [int]$shape.Range.Information(3)
            width_pt = [math]::Round($shape.Width, 2)
            height_pt = [math]::Round($shape.Height, 2)
            type = [int]$shape.Type
            alt_text = [string]$shape.AlternativeText
        }
    }

    $result = [ordered]@{
        source = $InputDocx
        pages = [int]$document.ComputeStatistics(2)
        words = [int]$document.ComputeStatistics(0)
        paragraphs_total = $document.Paragraphs.Count
        tables_total = $document.Tables.Count
        inline_shapes_total = $document.InlineShapes.Count
        sections = $sections
        paragraphs = $paragraphs
        tables = $tables
        inline_shapes = $shapes
    }

    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputJson -Encoding UTF8
    $document.ExportAsFixedFormat($OutputPdf, 17)
}
finally {
    if ($document) { $document.Close(0) }
    if ($word) { $word.Quit() }
    [gc]::Collect()
    [gc]::WaitForPendingFinalizers()
}
