#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a static HTML gallery for MMBench results by category.

Features
- Groups related questions sharing the same base image id (e.g., 241 / 1000241 / 2000241 / 3000241).
- Per group shows: Question (Excel B), Options (D–G), GT Answer (H), Model Answer (N).
- Selects the first 20 image-groups per category (stable order).
- Copies images into output folder (unless --no-copy), producing a ready-to-publish site (index.html).

Usage
python gen_mmbench_site.py \
  --xlsx /pfs/lichenyi/station/VLMEval/outputs/mmbench_dev_en/UnifyModelEval/train_qwendit_unify_interleave_stage1p5/0000050000/UnifyModelEval_MMBench_DEV_EN.xlsx \
  --image-root /pfs/shared_eval/datasets/images/MMBench \
  --out-dir ./site \
  --image-prefix assets/mmbench

# Debug without copying images (keep absolute /pfs paths in <img src>):
python gen_mmbench_site.py \
  --xlsx /pfs/.../UnifyModelEval_MMBench_DEV_EN.xlsx \
  --image-root /pfs/shared_eval/datasets/images/MMBench \
  --out-dir ./site \
  --image-prefix /pfs/shared_eval/datasets/images/MMBench \
  --no-copy
"""

import argparse
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json


# ------------------------------- helpers ------------------------------------ #
def infer_columns(df: pd.DataFrame):
    cols = [str(c).strip() for c in df.columns]

    # Excel positions (fallback-friendly):
    # B -> question
    question_col = cols[1] if len(cols) > 1 else cols[0]
    # D-G -> options
    opt_cols = [cols[i] for i in [3, 4, 5, 6] if i < len(cols)]
    # H -> GT answer
    gt_col = cols[7] if len(cols) > 7 else None
    # N -> model answer
    model_col = cols[13] if len(cols) > 13 else cols[-1]

    # ID column candidates
    id_candidates = [c for c in cols if str(c).lower() in ["index", "id", "mmbench_id", "qid", "question_id"]]
    idx_col = id_candidates[0] if id_candidates else cols[0]

    # Category column
    cat_candidates = [c for c in cols if str(c).lower() in ["category", "cat", "task", "subtask"]]
    if cat_candidates:
        cat_col = cat_candidates[0]
    else:
        cat_subs = [c for c in cols if "cat" in str(c).lower() or "Category" in str(c)]
        cat_col = cat_subs[0] if cat_subs else None

    return {
        "idx": idx_col,
        "question": question_col,
        "options": opt_cols,
        "gt": gt_col,
        "model": model_col,
        "category": cat_col,
    }


def to_int_or_nan(x):
    try:
        return int(str(x).strip().split(".")[0])
    except Exception:
        return np.nan


def load_and_group(xlsx_path: Path):
    df = pd.read_excel(xlsx_path, sheet_name=0, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    cols = infer_columns(df)

    essentials = [cols["idx"], cols["question"], cols["model"]] + cols["options"]
    if cols["gt"] and cols["gt"] in df.columns:
        essentials.append(cols["gt"])
    essentials = [c for c in essentials if c in df.columns]

    cat_col = cols["category"]
    if cat_col and cat_col not in df.columns:
        cat_col = None

    use_df = df[essentials + ([cat_col] if cat_col else [])].copy()
    use_df["mmbench_id"] = use_df[cols["idx"]].apply(to_int_or_nan)
    use_df["base_img_id"] = use_df["mmbench_id"] % 1_000_000
    use_df["category"] = use_df[cat_col].astype(str) if cat_col else "Unknown"
    use_df.sort_values(["category", "base_img_id", "mmbench_id"], inplace=True, kind="mergesort")
    return use_df, cols


def pick_twenty_per_category(use_df: pd.DataFrame):
    selected = []
    for cat, sub in use_df.groupby("category", sort=False):
        base_ids = sub["base_img_id"].dropna().astype(int).unique().tolist()
        for b in base_ids[:20]:
            g = sub[sub["base_img_id"] == b].copy()
            selected.append(((cat, int(b)), g))
    return selected


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_images(selected_groups, image_root: Path, out_dir: Path, image_prefix: str, do_copy=True):
    """
    Returns map: base_id -> relative img src used in HTML.
    """
    copied_paths = {}
    dest_root = out_dir / image_prefix
    if do_copy:
        ensure_dir(dest_root)

    for (cat, base_id), _ in selected_groups:
        src = image_root / f"{base_id}.jpg"
        rel_dest = Path(image_prefix) / f"{base_id}.jpg"
        if do_copy:
            if src.exists():
                shutil.copy2(src, dest_root / f"{base_id}.jpg")
        copied_paths[base_id] = str(rel_dest).replace("\\", "/")
    return copied_paths


def html_escape(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_html(selected_groups, cols, img_src_map, site_title="MMBench Results Gallery"):
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head><meta charset='utf-8'/>")
    html.append(f"<title>{site_title}</title>")
    html.append(
        """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin:0; padding:0; background:#f7f7f9; }
header { position: sticky; top: 0; background: #fff; padding: 16px 24px; border-bottom: 1px solid #e5e7eb; z-index: 10; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.category { margin: 32px 0; }
.group-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; margin: 16px 0 32px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.group-head { display: flex; align-items: center; gap: 16px; padding: 12px 16px; border-bottom: 1px solid #f1f5f9; background:#fcfcfd; }
.group-head h3 { margin: 0; font-size: 18px; }
.group-body { display: flex; gap: 16px; padding: 16px; }
.group-img { width: 280px; flex: 0 0 280px; background: #fafafa; text-align: center; }
.group-img img { max-width: 100%; height: auto; display: block; }
.qa-table { width: 100%; border-collapse: collapse; }
.qa-table th, .qa-table td { border-bottom: 1px solid #f1f5f9; text-align: left; padding: 8px 10px; vertical-align: top; }
.qa-table th { background: #fafafa; font-weight: 600; }
.badge { display: inline-block; padding: 2px 8px; font-size: 12px; border-radius: 999px; background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.cat-nav { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.cat-nav a { text-decoration: none; padding: 6px 10px; border: 1px solid #e5e7eb; background: #fff; border-radius: 999px; color: #374151; font-size: 14px; }
.cat-nav a:hover { background: #f9fafb; }
.small { color: #6b7280; font-size: 12px; }
footer { color: #6b7280; font-size: 12px; padding: 24px; text-align: center; }
</style>
        """
    )
    html.append("</head><body>")

    # Collect category order
    cat_order = []
    for (cat, _), _g in selected_groups:
        if cat not in cat_order:
            cat_order.append(cat)

    # Header with category nav
    html.append("<header>")
    html.append(f"<h1 style='margin:0'>{html_escape(site_title)}</h1>")
    html.append("<div class='small'>Shows per category the first 20 image-groups, grouping related questions (e.g., 241 / 1000241 / 2000241 / 3000241).</div>")
    html.append("<div class='cat-nav'>")
    for c in cat_order:
        anchor = c.lower().replace(" ", "-")
        html.append(f"<a href='#{anchor}'>{html_escape(c)}</a>")
    html.append("</div>")
    html.append("</header>")

    # Body
    html.append("<div class='container'>")
    by_cat = {}
    for (cat, base_id), g in selected_groups:
        by_cat.setdefault(cat, []).append((base_id, g))

    q_col = cols["question"]
    opt_cols = cols["options"]
    gt_col = cols["gt"]
    model_col = cols["model"]

    for c in cat_order:
        anchor = c.lower().replace(" ", "-")
        html.append(f"<div class='category' id='{anchor}'>")
        html.append(f"<h2>{html_escape(c)}</h2>")

        for base_id, g in by_cat.get(c, []):
            img_src = img_src_map.get(base_id, f"{base_id}.jpg")
            html.append("<section class='group-card'>")
            html.append("<div class='group-head'>")
            html.append(f"<span class='badge'>Image {base_id}.jpg</span>")
            html.append(f"<h3>Grouped QAs for Base ID {base_id}</h3>")
            html.append("</div>")

            html.append("<div class='group-body'>")
            html.append(f"<div class='group-img'><img alt='{base_id}.jpg' src='{html_escape(img_src)}'/></div>")

            # QA table
            html.append("<div style='flex:1'>")
            html.append("<table class='qa-table'>")
            html.append("<thead><tr>")
            html.append("<th>MMBench ID</th>")
            html.append("<th>Question (B)</th>")
            html.append("<th>A (D)</th><th>B (E)</th><th>C (F)</th><th>D (G)</th>")
            html.append("<th>GT Answer (H)</th>")
            html.append("<th>Model Answer (N)</th>")
            html.append("</tr></thead><tbody>")

            for _, r in g.iterrows():
                mmid = r.get("mmbench_id", "")
                q = r.get(q_col, "")

                def gopt(i):
                    return r.get(opt_cols[i], "") if i < len(opt_cols) else ""

                oA, oB, oC, oD = gopt(0), gopt(1), gopt(2), gopt(3)
                gt = r.get(gt_col, "") if gt_col else ""
                md = r.get(model_col, "")

                html.append("<tr>")
                html.append(f"<td>{html_escape(mmid)}</td>")
                html.append(f"<td>{html_escape(q)}</td>")
                html.append(f"<td>{html_escape(oA)}</td>")
                html.append(f"<td>{html_escape(oB)}</td>")
                html.append(f"<td>{html_escape(oC)}</td>")
                html.append(f"<td>{html_escape(oD)}</td>")
                html.append(f"<td>{html_escape(gt)}</td>")
                html.append(f"<td>{html_escape(md)}</td>")
                html.append("</tr>")

            html.append("</tbody></table>")
            html.append("</div>")   # flex
            html.append("</div>")   # group-body
            html.append("</section>")

        html.append("</div>")  # category
    html.append("</div>")      # container
    html.append("<footer>Generated by gen_mmbench_site.py</footer>")
    html.append("</body></html>")
    return "\n".join(html)


# ---------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to MMBench Excel file")
    ap.add_argument("--image-root", required=True, help="Root folder of MMBench images")
    ap.add_argument("--out-dir", default="./site", help="Output directory for the site")
    ap.add_argument("--image-prefix", default="assets/mmbench", help="Relative path for images inside out-dir")
    ap.add_argument("--no-copy", action="store_true", help="If set, do not copy images; keep <img src> pointing to image-prefix as-is")
    ap.add_argument("--site-title", default="MMBench Results Gallery", help="HTML <title>")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    image_prefix = args.image_prefix

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load, group, select
    use_df, cols = load_and_group(xlsx_path)
    selected_groups = pick_twenty_per_category(use_df)

    # Map or copy images
    img_src_map = copy_images(
        selected_groups,
        image_root,
        out_dir,
        image_prefix,
        do_copy=(not args.no_copy)
    )

    # Render HTML
    html = build_html(selected_groups, cols, img_src_map, site_title=args.site_title)
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")

    # Save manifest for downstream tooling
    manifest = []
    for (cat, base_id), g in selected_groups:
        manifest.append({
            "category": cat,
            "base_img_id": int(base_id),
            "image": img_src_map.get(base_id, f"{base_id}.jpg"),
            "mmbench_ids": [int(i) for i in g["mmbench_id"].tolist() if pd.notna(i)]
        })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote: {index_path}")
    print(f"[OK] Wrote: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
