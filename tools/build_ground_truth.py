"""Build evaluation/ground_truth.json from the text layer of the specimen PDF.

For every one of the 80 pages (10 patients x 8 page types) we know exactly what was
handwritten where (handwriting fonts) and which boxes are ticked (coloured strokes), so we
can score the extractor field by field without labelling anything by hand.

Direct identifiers (pii fields) are skipped: their values are never written anywhere.

    python -m tools.build_ground_truth
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pymupdf

from extraction.catalog import CATALOG, PAGE_ORDER
from extraction.geometry import apply, contains, fit_affine, intersects, invert, map_rect
from extraction.pdf_layout import chars_to_text, read_page
from extraction.template import load_templates
from tools.paths import EVAL_DIR, GROUND_TRUTH, REGISTRY, SPECIMEN_PDF

DASH = re.compile(r"^[\s\-–—_]*$")
# The NanumPen handwriting font used for some patients has no glyph for accented letters
# or "—": the PDF records U+FFFD and the rendered image shows a gap.
MISSING = "\ufffd"


def _c(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def label_affine(tpl, layout):
    """Affine template -> page from printed labels.

    Fit on labels that occur once on both pages, then add repeated labels ("Visite 1",
    "Normal"...) by pairing each with its nearest counterpart under that first fit."""
    seen_t, seen_p = defaultdict(list), defaultdict(list)
    for l in tpl.labels:
        seen_t[l["text"]].append(_c(l["bbox"]))
    for l in layout.labels:
        seen_p[l.text].append(_c(l.bbox))
    uniq = [(seen_t[t][0], seen_p[t][0]) for t in seen_t if len(seen_t[t]) == 1 and len(seen_p.get(t, [])) == 1]
    M = fit_affine(np.array([u[0] for u in uniq]), np.array([u[1] for u in uniq]))
    src, dst = [u[0] for u in uniq], [u[1] for u in uniq]
    for t, tpts in seen_t.items():
        ppts = seen_p.get(t, [])
        if len(tpts) > 1 and len(ppts) == len(tpts):
            for tp, q in zip(tpts, apply(M, tpts)):
                j = int(np.argmin([np.hypot(*(np.array(p) - q)) for p in ppts]))
                src.append(tp)
                dst.append(ppts[j])
    M = fit_affine(np.array(src), np.array(dst))
    resid = np.abs(apply(M, src) - np.array(dst)).max()
    return M, resid


def png_index():
    idx = defaultdict(list)
    for p in sorted(REGISTRY.glob("dossiers_specimen_10_patientes-*.png")):
        m = re.match(r".*-(\d+)(?:__.*)?\.png$", p.name)
        idx[int(m.group(1))].append(p.name)
    return idx


def page_truth(page, page_type, tpl):
    layout = read_page(page)
    M, resid = label_affine(tpl, layout)
    Minv = invert(M)
    fields, owners = {}, {}

    # handwritten groups -> text/cell fields (by where each group starts, in template coords)
    by_field = defaultdict(list)
    unassigned = []
    for g in layout.hw:
        x, y = apply(Minv, [(g.bbox[0] + 1.0, g.yc)])[0]
        owner = None
        for f in CATALOG[page_type]:
            geo = tpl.fields.get(f.key)
            if geo and "region" in geo and contains(geo["region"], x, y):
                owner = f
                break
        if owner is None:
            unassigned.append(g)
        else:
            by_field[owner.key].append(g)
            owners[owner.key] = owner

    for f in CATALOG[page_type]:
        if f.pii or f.key not in tpl.fields:
            continue
        geo = tpl.fields[f.key]
        if f.kind in ("text", "cell"):
            gs = sorted(by_field.get(f.key, []), key=lambda g: g.order)
            raw = chars_to_text([c for g in gs for c in g.chars])
            if not raw:
                fields[f.key] = {"raw": None, "status": "NOT_PROVIDED"}
            elif DASH.match(raw.replace(MISSING, "")):
                # "—"; when the font lacked the dash glyph nothing is visible on the image
                fields[f.key] = {"raw": "—", "status": "NOT_APPLICABLE"}
                if MISSING in raw:
                    fields[f.key]["glyph_missing"] = True
            else:
                fields[f.key] = {"raw": raw, "status": "KNOWN"}
                if MISSING in raw:
                    fields[f.key]["glyph_missing"] = True
        elif f.kind == "check":
            box = map_rect(M, geo["box"])
            ticked = any(intersects(box, m, pad=1.0) for m in layout.marks)
            fields[f.key] = {"value": ticked, "status": "KNOWN"}
        elif f.kind == "radio":
            chosen = [v for v, b in geo["options"].items()
                      if any(intersects(map_rect(M, b), m, pad=1.0) for m in layout.marks)]
            if not chosen:
                fields[f.key] = {"value": None, "status": "NOT_PROVIDED"}
            elif len(chosen) == 1:
                fields[f.key] = {"value": chosen[0], "status": "KNOWN"}
            else:
                fields[f.key] = {"value": chosen, "status": "NEEDS_REVIEW"}

    pii_regions = [tpl.fields[f.key]["region"] for f in CATALOG[page_type] if f.pii and f.key in tpl.fields]
    leftovers = []
    for g in layout.hw:
        if any(g is u for u in unassigned):
            x, y = apply(Minv, [(g.bbox[0] + 1.0, g.yc)])[0]
            if not any(contains(r, x, y) for r in pii_regions):
                leftovers.append(f"{g.text!r}@({x:.0f},{y:.0f})")
    return fields, resid, leftovers


def main():
    doc = pymupdf.open(SPECIMEN_PDF)
    templates = load_templates()
    pngs = png_index()
    pages, n_left = {}, 0
    for i, page in enumerate(doc):
        page_type = PAGE_ORDER[i % 8]
        patient = i // 8 + 1
        fields, resid, leftovers = page_truth(page, page_type, templates[page_type])
        pages[str(i + 1)] = {"patient": patient, "page_type": page_type, "images": pngs.get(i + 1, []),
                             "fields": fields}
        if leftovers or resid > 0.5:
            print(f"page {i + 1:2d} {page_type:26s} affine resid {resid:.2f}  unassigned: {', '.join(leftovers)}")
        n_left += len(leftovers)

    EVAL_DIR.mkdir(exist_ok=True)
    out = {"source": SPECIMEN_PDF.name,
           "note": "Generated by tools/build_ground_truth.py from the PDF text layer. Identifiers excluded.",
           "pages": pages}
    GROUND_TRUTH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = defaultdict(int)
    for p in pages.values():
        for v in p["fields"].values():
            counts[v["status"]] += 1
    print(f"wrote {GROUND_TRUTH}: {len(pages)} pages, {sum(counts.values())} fields {dict(counts)}; "
          f"{n_left} handwritten groups not assigned to any field")


if __name__ == "__main__":
    main()
