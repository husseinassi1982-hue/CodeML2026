"""Write real extractor outputs to fixtures/ so the chat UI and backend can be built and
tested without running OCR:

    python -m tools.make_fixtures

fixtures/<page_type>.json        full PageExtraction (what the backend receives)
fixtures/<page_type>.ui.json     {key: {label, value, conf, status}} for the chat mock-up
fixtures/<page_type>_photo.json  same page through the simulated phone-photo degradation
fixtures/manual_<page_type>.json empty form for full manual entry (AI unavailable)
"""

import json

import cv2

from extraction import PAGE_TYPES, empty_form, extract
from tools.degrade import degrade
from tools.paths import GROUND_TRUTH, REGISTRY, REPO

OUT = REPO / "fixtures"


def main():
    OUT.mkdir(exist_ok=True)
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["pages"]
    first = {}
    for pg, info in gt.items():  # patient 1's pages
        if info["patient"] == 1:
            first[info["page_type"]] = REGISTRY / info["images"][0]
    for page_type, path in first.items():
        r = extract(path)
        (OUT / f"{page_type}.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
        (OUT / f"{page_type}.ui.json").write_text(json.dumps(r.to_ui_fields(), ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        print(f"{page_type:26s} {r.summary}")
    for page_type in ("current_pregnancy", "cover"):
        r = extract(degrade(cv2.imread(str(first[page_type])), "medium", seed=7))
        (OUT / f"{page_type}_photo.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
        print(f"{page_type + ' (photo)':26s} {r.summary}")
    for page_type in PAGE_TYPES:
        (OUT / f"manual_{page_type}.json").write_text(empty_form(page_type).model_dump_json(indent=2),
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
