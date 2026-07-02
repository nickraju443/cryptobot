"""Convert STRATEGY.md to a styled HTML page, then to PDF via Chrome headless."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
MD = HERE / "STRATEGY.md"
HTML = HERE / "STRATEGY.html"
PDF = HERE / "STRATEGY.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #1f2937;
  max-width: 100%;
  margin: 0;
  padding: 0;
}
h1 {
  font-size: 22pt;
  color: #0f172a;
  border-bottom: 3px solid #f59e0b;
  padding-bottom: 6px;
  margin-top: 0;
  page-break-after: avoid;
}
h2 {
  font-size: 16pt;
  color: #0f172a;
  margin-top: 26px;
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 4px;
  page-break-after: avoid;
}
h3 {
  font-size: 13pt;
  color: #1f2937;
  margin-top: 18px;
  page-break-after: avoid;
}
p, ul, ol, table { page-break-inside: avoid; }
ul, ol { padding-left: 22px; }
li { margin: 2px 0; }
code {
  font-family: "JetBrains Mono", "Cascadia Mono", Consolas, monospace;
  font-size: 10pt;
  background: #f1f5f9;
  padding: 1px 5px;
  border-radius: 3px;
  color: #b91c1c;
}
pre {
  background: #0f172a;
  color: #e2e8f0;
  padding: 12px 14px;
  border-radius: 6px;
  font-family: "JetBrains Mono", Consolas, monospace;
  font-size: 9.5pt;
  line-height: 1.4;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
  font-size: 10pt;
}
th, td {
  border: 1px solid #e5e7eb;
  padding: 6px 9px;
  text-align: left;
}
th {
  background: #f8fafc;
  font-weight: 600;
  color: #0f172a;
}
tr:nth-child(even) td { background: #fafafa; }
hr {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 18px 0;
}
strong { color: #0f172a; }
blockquote {
  border-left: 3px solid #f59e0b;
  padding: 4px 12px;
  margin: 10px 0;
  color: #475569;
  background: #fffbeb;
}
.header-meta {
  color: #6b7280;
  font-size: 10pt;
  margin-bottom: 18px;
}
"""

def md_to_html():
    import markdown
    md_text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>CryptoBot — Trading Strategy</title>
<style>{CSS}</style>
</head><body>
<div class="header-meta">Generated from STRATEGY.md · CryptoBot — 24/7 crypto scalper</div>
{body}
</body></html>"""
    HTML.write_text(html, encoding="utf-8")
    print(f"HTML written -> {HTML}")

def html_to_pdf():
    # Use Chrome headless to render the HTML to PDF
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not Path(chrome).exists():
        chrome = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    file_url = "file:///" + str(HTML).replace("\\", "/")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        file_url,
    ]
    print(f"Running Chrome headless ...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print("Chrome stderr:", r.stderr[:800])
        sys.exit(1)
    if not PDF.exists() or PDF.stat().st_size < 1000:
        print(f"PDF not produced or too small: {PDF}")
        sys.exit(1)
    print(f"PDF written -> {PDF} ({PDF.stat().st_size:,} bytes)")

def open_pdf():
    print(f"Opening {PDF} ...")
    os.startfile(str(PDF))

if __name__ == "__main__":
    md_to_html()
    html_to_pdf()
    open_pdf()
