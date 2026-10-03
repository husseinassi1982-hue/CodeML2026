"""python -m extraction PHOTO [--backend local|claude] [--ui]"""

import argparse
import json
import sys

from . import ExtractionError, extract


def main():
    ap = argparse.ArgumentParser(description="Extract structured fields from a registry page photo.")
    ap.add_argument("image")
    ap.add_argument("--backend", default="local", choices=["local", "claude"])
    ap.add_argument("--ui", action="store_true", help="print the {label, value, conf, status} shape used by the chat UI")
    ap.add_argument("--review", action="store_true", help="print only the fields that need review")
    a = ap.parse_args()
    try:
        r = extract(a.image, backend=a.backend)
    except ExtractionError as e:
        print(json.dumps({"error": e.code, "message": str(e)}), file=sys.stderr)
        sys.exit(1)
    if a.ui:
        print(json.dumps(r.to_ui_fields(r.needs_review if a.review else None), ensure_ascii=False, indent=2))
    elif a.review:
        print(json.dumps({k: r.fields[k].model_dump(mode="json") for k in r.needs_review}, ensure_ascii=False, indent=2))
    else:
        print(r.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
