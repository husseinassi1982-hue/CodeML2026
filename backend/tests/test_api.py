"""End-to-end through the HTTP API, using saved real extractor outputs (no OCR needed)."""
import importlib.util

import pytest


def capture(client, photo, page="cover_photo"):
    r = client.post("/records", files={"photo": photo()}, data={"demo_page": page})
    assert r.status_code == 201, r.text
    return r.json()


def test_health_and_front_end(client):
    assert client.get("/health").json()["status"] == "healthy"
    page = client.get("/")
    assert page.status_code == 200 and "DayOne" in page.text


def test_full_visit_cover_page(client, photo):
    rec = capture(client, photo)
    assert rec["state"] == "PENDING_AI"

    report = client.post("/queue/process").json()
    assert report["processed"] == [rec["id"]]

    rec = client.get(f"/records/{rec['id']}").json()
    assert rec["state"] == "TO_REVIEW"
    assert rec["page_type"] == "cover"
    keys = [f["key"] for f in rec["needs_review"]]
    assert "record_number" in keys and "facility_name" in keys
    assert rec["needs_review"][0]["question_en"]

    # midwife confirms the code and corrects the facility name
    r = client.patch(f"/records/{rec['id']}/fields", json={"answers": {
        "record_number": {"confirm": True}, "facility_name": {"text": "CSCA Al Wifaq"}}})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["needs_review"] == []
    assert rec["record_number"] == "2026-823-001"

    assert client.post(f"/records/{rec['id']}/validate").json()["state"] == "VALIDATED"
    r = client.post(f"/records/{rec['id']}/link", json={"patient_code": rec["record_number"], "create": True})
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "SAVED" and r.json()["patient_code"] == "2026-823-001"

    assert client.post("/sync").json()["synced"] == [rec["id"]]
    full = client.get(f"/records/{rec['id']}").json()
    assert full["state"] == "SYNCED"
    states = [h["new_state"] for h in full["history"]]
    assert states == ["CAPTURED", "PENDING_AI", "AI_PROCESSED", "TO_REVIEW", "VALIDATED",
                      "PATIENT_LINKED", "SAVED", "SYNCED"]


def test_typed_values_are_parsed_and_markers_understood(client, photo):
    rec = capture(client, photo, "current_pregnancy_photo")
    client.post("/queue/process")
    rec = client.get(f"/records/{rec['id']}").json()
    bp = next(f for f in rec["fields"].values() if f["dtype"] == "bp")
    unk = next(f for f in rec["fields"].values() if f["dtype"] == "int" and f["key"] != bp["key"])
    r = client.patch(f"/records/{rec['id']}/fields", json={"answers": {
        bp["key"]: {"text": "120/80"}, unk["key"]: {"text": "?"}}}).json()
    full = client.get(f"/records/{rec['id']}").json()["fields"]
    assert full[bp["key"]]["status"] == "KNOWN" and full[bp["key"]]["corrected_by_midwife"]
    assert full[bp["key"]]["value"] == {"systolic": 120, "diastolic": 80}
    assert full[unk["key"]]["status"] == "UNKNOWN"
    assert bp["key"] not in [f["key"] for f in r["needs_review"]]


def test_offline_capture_then_sync_waits_for_connection(client, photo):
    client.post("/connectivity", json={"online": False})
    rec = capture(client, photo)
    client.post("/queue/process")  # local AI works offline
    rid = rec["id"]
    client.patch(f"/records/{rid}/fields", json={"answers": {"record_number": {"confirm": True},
                                                             "facility_name": {"confirm": True}}})
    client.post(f"/records/{rid}/validate")
    client.post(f"/records/{rid}/link", json={"create": True})
    assert client.post("/sync").json()["waiting"] == [rid]
    assert client.get("/status").json()["counts"]["SAVED"] == 1
    client.post("/connectivity", json={"online": True})
    assert client.post("/sync").json()["synced"] == [rid]


def test_ai_waits_for_network_when_configured(client, photo, monkeypatch):
    from core import config
    monkeypatch.setattr(config, "AI_NEEDS_NETWORK", True)
    client.post("/connectivity", json={"online": False})
    rec = capture(client, photo)
    assert client.post("/queue/process").json()["waiting"] == [rec["id"]]
    client.post("/connectivity", json={"online": True})
    assert client.post("/queue/process").json()["processed"] == [rec["id"]]


def test_same_photo_twice_is_flagged(client):
    f = ("p.jpg", b"identical", "image/jpeg")
    client.post("/records", files={"photo": f}, data={"demo_page": "cover"})
    second = client.post("/records", files={"photo": f}, data={"demo_page": "cover"}).json()
    assert second["state"] == "SUSPECTED_DUPLICATE"


def test_linking_never_guesses_a_patient(client, photo):
    first = capture(client, photo)
    client.post("/queue/process")
    client.patch(f"/records/{first['id']}/fields", json={"answers": {"record_number": {"confirm": True},
                                                                     "facility_name": {"confirm": True}}})
    client.post(f"/records/{first['id']}/validate")
    client.post(f"/records/{first['id']}/link", json={"patient_code": "2026-823-001", "create": True})

    second = capture(client, photo, "current_pregnancy")
    client.post("/queue/process")
    rid = second["id"]
    key = client.get(f"/records/{rid}").json()["needs_review"][0]["key"]
    client.patch(f"/records/{rid}/fields", json={"answers": {key: {"confirm": True}}})
    client.post(f"/records/{rid}/validate")

    # one digit off: refused, with the near match offered
    r = client.post(f"/records/{rid}/link", json={"patient_code": "2026-823-007"})
    assert r.status_code == 404
    assert [c["code"] for c in r.json()["candidates"]] == ["2026-823-001"]
    assert client.get("/patients", params={"code": "2026-832-001"}).json()[0]["code"] == "2026-823-001"

    r = client.post(f"/records/{rid}/link", json={"patient_code": "2026-823-001"})
    assert r.status_code == 200
    p = client.get("/patients/2026-823-001").json()
    assert p["visits"] == 2 and {x["page_type"] for x in p["records"]} == {"cover", "current_pregnancy"}


def test_illegal_moves_are_refused(client, photo):
    rec = capture(client, photo)
    assert client.post(f"/records/{rec['id']}/validate").status_code == 409  # not reviewed yet
    assert client.patch(f"/records/{rec['id']}/fields",
                        json={"answers": {"x": {"confirm": True}}}).status_code == 409
    client.post("/queue/process")
    assert client.patch(f"/records/{rec['id']}/fields",
                        json={"answers": {"not_a_field": {"confirm": True}}}).status_code == 422
    assert client.post(f"/records/{rec['id']}/link", json={"create": True}).status_code == 409
    assert client.get("/records/nope").status_code == 404
    assert client.patch(f"/records/{rec['id']}/fields",
                        json={"answers": {"region": {"confirm": True, "text": "x"}}}).status_code == 422
    assert client.post("/records", files={"photo": photo()}, data={"demo_page": "bogus"}).status_code == 422


def test_photo_access_is_restricted(client, photo):
    rec = capture(client, photo)
    assert client.get(f"/records/{rec['id']}/image").status_code == 403
    assert client.get(f"/records/{rec['id']}/image", headers={"X-Role": "researcher"}).status_code == 403
    r = client.get(f"/records/{rec['id']}/image", headers={"X-Role": "midwife"})
    assert r.status_code == 200 and r.content.startswith(b"fake-jpeg")


def test_extractor_errors_map_to_states(client, photo, monkeypatch):
    from services import ai_service

    monkeypatch.setattr(ai_service, "active_extractor", lambda: "local")
    import extraction

    def unreadable(img, backend="local"):
        raise extraction.ExtractionError("unreadable_image", "not an image")
    monkeypatch.setattr(extraction, "extract", unreadable)
    rec = client.post("/records", files={"photo": photo()}).json()
    assert client.post("/queue/process").json()["failed"] == [rec["id"]]  # immediately, no 3 retries
    assert client.get(f"/records/{rec['id']}").json()["state"] == "PROCESSING_FAILED"

    # midwife gives up on the photo and types the page herself
    r = client.post(f"/records/{rec['id']}/manual", json={"page_type": "cover"})
    assert r.status_code == 200 and r.json()["state"] == "MANUAL_REVIEW_REQUIRED"
    r = client.patch(f"/records/{rec['id']}/fields", json={"answers": {"region": {"text": "Rabat-Salé-Kénitra"}}})
    assert r.json()["state"] == "TO_REVIEW"


def test_unrecognised_page_goes_to_manual_review(client, photo, monkeypatch):
    from services import ai_service
    from extraction import empty_form

    monkeypatch.setattr(ai_service, "active_extractor", lambda: "local")
    import extraction
    unknown = empty_form("cover").model_copy(update={"page_type": "unknown"})
    monkeypatch.setattr(extraction, "extract", lambda img, backend="local": unknown)
    rec = client.post("/records", files={"photo": photo()}).json()
    client.post("/queue/process")
    assert client.get(f"/records/{rec['id']}").json()["state"] == "MANUAL_REVIEW_REQUIRED"


@pytest.mark.skipif(importlib.util.find_spec("rapidocr") is None, reason="OCR not installed")
def test_real_ocr_on_printed_cover(client):
    from core import config
    path = config.REPO_ROOT / "data" / "Paper Registry" / "dossiers_specimen_10_patientes-01.png"
    if not path.exists():
        pytest.skip("organisers' data not linked (ln -s ../dayone-participants/data data)")
    img = path.read_bytes()
    rec = client.post("/records", files={"photo": ("cover.png", img, "image/png")}).json()
    report = client.post("/queue/process").json()
    assert report["processed"] == [rec["id"]]
    assert client.get(f"/records/{rec['id']}").json()["page_type"] == "cover"


def test_typed_answers_must_fit_the_field(client, photo):
    rec = capture(client, photo, "current_pregnancy_photo")
    client.post("/queue/process")
    fields = client.get(f"/records/{rec['id']}").json()["fields"]
    posneg = next(k for k, f in fields.items() if f["dtype"] == "posneg")
    date = next(k for k, f in fields.items() if f["dtype"] == "date")

    def send(key, text):
        return client.patch(f"/records/{rec['id']}/fields", json={"answers": {key: {"text": text}}})

    r = send(posneg, "120/80")
    assert r.status_code == 422 and "120/80" in r.json()["detail"]
    assert send(date, "tomorrow").status_code == 422
    assert send(posneg, "Neg").json()["state"] == "TO_REVIEW"
    assert send(date, "18/04/2026").status_code == 200
    # a rejected batch saves nothing
    before = client.get(f"/records/{rec['id']}").json()["fields"][date]
    assert client.patch(f"/records/{rec['id']}/fields", json={"answers": {
        date: {"text": "01/01/2026"}, posneg: {"text": "nonsense"}}}).status_code == 422
    assert client.get(f"/records/{rec['id']}").json()["fields"][date]["value"] == before["value"]


def test_checkbox_and_choice_answers(client, photo):
    rec = capture(client, photo)
    client.post("/queue/process")
    url = f"/records/{rec['id']}/fields"
    assert client.patch(url, json={"answers": {"high_risk": {"text": "oui"}}}).status_code == 200
    assert client.patch(url, json={"answers": {"high_risk": {"text": "maybe"}}}).status_code == 422
    r = client.patch(url, json={"answers": {"facility_type": {"text": "banana"}}})
    assert r.status_code == 422 and "CSCA" in r.json()["detail"]
    f = client.get(f"/records/{rec['id']}").json()["fields"]
    assert f["high_risk"]["value"] is True
