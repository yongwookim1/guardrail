#!/bin/bash
# visualization.sh — Generate VLSU comparison HTML
# Usage: bash visualization.sh [output.html]
set -e

GUARDREASONER=$(cd "$(dirname "$0")" && pwd)
LLAMA_FACTORY=${LLAMA_FACTORY:-$(cd "$GUARDREASONER/.." && pwd)/LLaMA-Factory}
OUTPUT_HTML=${1:-"$GUARDREASONER/vlsu_visualization.html"}

echo "Building VLSU visualization..."
echo "  data dir : $GUARDREASONER/data/test"
echo "  image dir: $LLAMA_FACTORY/data/vlsu_image"
echo "  output   : $OUTPUT_HTML"

python3 - "$GUARDREASONER" "$LLAMA_FACTORY" "$OUTPUT_HTML" <<'PY'
import json
import sys
import base64
import re
from pathlib import Path
from collections import defaultdict

guardreasoner = Path(sys.argv[1])
llama_factory  = Path(sys.argv[2])
output_html    = Path(sys.argv[3])

TEST_DIR = guardreasoner / "data" / "test"

# Image search dirs (in priority order)
IMAGE_DIRS = [
    llama_factory / "data" / "vlsu_image",
    guardreasoner / "train" / "vlsu_image",
    llama_factory / "vlsu_image",
]

def find_image(rel_path: str) -> Path | None:
    fname = Path(rel_path).name
    for d in IMAGE_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None

def img_to_b64(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"

def extract_query(text_input: str) -> str:
    """Pull out just the human user text (strip role prefixes)."""
    lines = text_input.splitlines()
    collecting = False
    query_lines = []
    for line in lines:
        if line.startswith("Human user:"):
            collecting = True
            rest = line[len("Human user:"):].strip()
            if rest:
                query_lines.append(rest)
        elif line.startswith("AI assistant:"):
            break
        elif collecting:
            if line.strip() != "<image>":
                query_lines.append(line)
    return "\n".join(query_lines).strip()

def extract_ai_response(text_input: str) -> str:
    lines = text_input.splitlines()
    collecting = False
    res = []
    for line in lines:
        if line.startswith("AI assistant:"):
            collecting = True
            rest = line[len("AI assistant:"):].strip()
            if rest:
                res.append(rest)
        elif collecting:
            res.append(line)
    return "\n".join(res).strip() or "(no response)"

def parse_think_result(text_output: str):
    think = re.search(r"<think>(.*?)</think>", text_output, re.DOTALL)
    result = re.search(r"<result>(.*?)</result>", text_output, re.DOTALL)
    return (
        think.group(1).strip() if think else "",
        result.group(1).strip() if result else text_output.strip(),
    )

# Discover all VLSU prediction files
models = {}
for model_dir in sorted(TEST_DIR.iterdir()):
    pred_file = model_dir / "VLSU" / "generated_predictions.jsonl"
    if pred_file.exists():
        models[model_dir.name] = pred_file

if not models:
    print("No VLSU prediction files found under", TEST_DIR)
    sys.exit(1)

print(f"Found {len(models)} model(s): {list(models)}")

# Load all predictions indexed by uuid
all_data = defaultdict(dict)  # uuid -> {model_name: row}
for model_name, pred_file in models.items():
    with pred_file.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_data[row["uuid"]][model_name] = row

# Use first model's row ordering as canonical
first_model = next(iter(models))
uuids = [
    row["uuid"]
    for line in (models[first_model].open(encoding="utf-8"))
    for row in [json.loads(line)]
]

model_names = list(models.keys())

# Per-model stats
stats = {}
for m in model_names:
    rows = [all_data[u][m] for u in uuids if m in all_data[u]]
    correct = sum(1 for r in rows if r["predict"] == r["label"])
    stats[m] = {"total": len(rows), "correct": correct, "acc": 100 * correct / max(len(rows), 1)}

# HTML generation
BADGE_OK  = '<span class="badge ok">correct</span>'
BADGE_ERR = '<span class="badge err">wrong</span>'
LABEL_COLORS = {"harmful": "#e74c3c", "unharmful": "#27ae60"}

header_cells = "".join(
    f'<th>{m}<br><small>{stats[m]["correct"]}/{stats[m]["total"]} &nbsp; {stats[m]["acc"]:.1f}%</small></th>'
    for m in model_names
)

rows_html_parts = []
for uid in uuids:
    entry = all_data[uid]
    if not entry:
        continue
    ref = next(iter(entry.values()))
    query   = extract_query(ref["text_input"])
    ai_resp = extract_ai_response(ref["text_input"])
    label   = ref["label"]
    img_path = find_image(ref.get("image", ""))
    img_tag  = f'<img src="{img_to_b64(img_path)}" alt="image">' if img_path else '<div class="no-img">no image</div>'

    label_color = LABEL_COLORS.get(label, "#888")
    label_badge = f'<span class="label-tag" style="background:{label_color}">{label}</span>'

    any_wrong = any(
        entry[m]["predict"] != label for m in model_names if m in entry
    )
    row_cls = "row-wrong" if any_wrong else "row-ok"

    model_cells = []
    for m in model_names:
        if m not in entry:
            model_cells.append("<td><em>—</em></td>")
            continue
        r = entry[m]
        pred = r["predict"]
        think, result_text = parse_think_result(r.get("text_output", ""))
        correct = pred == label
        badge = BADGE_OK if correct else BADGE_ERR
        pred_color = LABEL_COLORS.get(pred, "#888")
        pred_span = f'<span style="color:{pred_color};font-weight:bold">{pred}</span>'
        think_html = (
            f'<details><summary>reasoning ({r.get("res_len",0)} chars)</summary>'
            f'<pre class="think">{think}</pre></details>'
            if think else ""
        )
        model_cells.append(
            f"<td>{badge} {pred_span}<br>{think_html}</td>"
        )

    category = ref.get("combined_category", "") or ""

    rows_html_parts.append(f"""
    <tr class="{row_cls}">
      <td class="img-cell">{img_tag}</td>
      <td class="query-cell">
        <div class="query">{query}</div>
        <div class="ai-resp"><b>AI resp:</b> {ai_resp}</div>
        <div class="meta">{label_badge} {f'<span class="cat">{category}</span>' if category else ''}</div>
      </td>
      {"".join(model_cells)}
    </tr>""")

rows_html = "\n".join(rows_html_parts)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VLSU Visualization</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; margin: 0; background: #f5f5f5; }}
  h1 {{ padding: 16px 24px; margin: 0; background: #2c3e50; color: #fff; font-size: 18px; }}
  .toolbar {{ padding: 10px 24px; background: #ecf0f1; border-bottom: 1px solid #ccc; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  .toolbar button {{ padding: 4px 12px; border: 1px solid #aaa; border-radius: 4px; cursor: pointer; background: #fff; }}
  .toolbar button.active {{ background: #2980b9; color: #fff; border-color: #2980b9; }}
  .stats {{ display: flex; gap: 24px; padding: 12px 24px; background: #fff; border-bottom: 1px solid #ddd; flex-wrap: wrap; }}
  .stat-box {{ background: #f0f4ff; border-radius: 6px; padding: 8px 16px; text-align: center; }}
  .stat-box .val {{ font-size: 22px; font-weight: bold; color: #2c3e50; }}
  .stat-box .lbl {{ font-size: 11px; color: #888; }}
  table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
  th {{ background: #2c3e50; color: #fff; padding: 8px 10px; text-align: left; position: sticky; top: 0; z-index: 10; }}
  td {{ padding: 8px 10px; vertical-align: top; border-bottom: 1px solid #e0e0e0; }}
  tr.row-wrong td {{ background: #fff5f5; }}
  tr.row-ok td {{ background: #fff; }}
  tr:hover td {{ filter: brightness(0.97); }}
  .img-cell {{ width: 160px; }}
  .img-cell img {{ width: 150px; height: 110px; object-fit: cover; border-radius: 4px; border: 1px solid #ccc; }}
  .no-img {{ width: 150px; height: 110px; background: #eee; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #aaa; font-size: 11px; }}
  .query-cell {{ width: 240px; }}
  .query {{ margin-bottom: 4px; color: #333; }}
  .ai-resp {{ font-size: 11px; color: #666; margin-bottom: 4px; }}
  .meta {{ margin-top: 4px; }}
  .label-tag {{ display: inline-block; padding: 2px 7px; border-radius: 10px; color: #fff; font-size: 11px; }}
  .cat {{ font-size: 10px; color: #888; background: #eee; padding: 1px 5px; border-radius: 8px; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 11px; color: #fff; }}
  .badge.ok {{ background: #27ae60; }}
  .badge.err {{ background: #e74c3c; }}
  pre.think {{ white-space: pre-wrap; font-size: 11px; background: #f8f8f8; padding: 8px; border-radius: 4px; border: 1px solid #eee; max-height: 300px; overflow-y: auto; }}
  details summary {{ cursor: pointer; font-size: 11px; color: #2980b9; }}
  #search {{ padding: 4px 8px; border: 1px solid #aaa; border-radius: 4px; width: 200px; }}
</style>
</head>
<body>
<h1>VLSU Model Comparison</h1>
<div class="stats">
{"".join(f'<div class="stat-box"><div class="val">{stats[m]["acc"]:.1f}%</div><div class="lbl">{m}<br>({stats[m]["correct"]}/{stats[m]["total"]})</div></div>' for m in model_names)}
  <div class="stat-box"><div class="val">{len(uuids)}</div><div class="lbl">total samples</div></div>
</div>
<div class="toolbar">
  <button class="active" onclick="filter('all', this)">All ({len(uuids)})</button>
  <button onclick="filter('wrong', this)">Any wrong</button>
  <button onclick="filter('ok', this)">All correct</button>
  <input id="search" type="text" placeholder="Search query..." oninput="filterSearch(this.value)">
</div>
<div style="overflow-x:auto">
<table>
  <thead><tr>
    <th style="width:160px">Image</th>
    <th style="width:240px">Query / Label</th>
    {header_cells}
  </tr></thead>
  <tbody id="tbody">
{rows_html}
  </tbody>
</table>
</div>
<script>
function filter(type, btn) {{
  document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#tbody tr').forEach(tr => {{
    if (type === 'all') tr.style.display = '';
    else if (type === 'wrong') tr.style.display = tr.classList.contains('row-wrong') ? '' : 'none';
    else if (type === 'ok') tr.style.display = tr.classList.contains('row-ok') ? '' : 'none';
  }});
}}
function filterSearch(q) {{
  const lq = q.toLowerCase();
  document.querySelectorAll('#tbody tr').forEach(tr => {{
    const text = tr.querySelector('.query')?.textContent?.toLowerCase() || '';
    tr.style.display = (!lq || text.includes(lq)) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

output_html.write_text(html, encoding="utf-8")
print(f"\nDone! {len(uuids)} samples, {len(model_names)} model(s)")
print(f"Output: {output_html}")
PY

echo ""
echo "Open in browser:"
echo "  open $OUTPUT_HTML   # macOS"
echo "  xdg-open $OUTPUT_HTML   # Linux"
