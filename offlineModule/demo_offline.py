"""Terminal demo of module C. Run:  python demo_offline.py

Shows: capture while OFFLINE -> saved locally -> connection returns -> queue drains
(and the connection dropping again in the middle of it, with nothing lost).
"""
import shutil
import time

from offline import Connectivity, RecordStore, State, process_queue

DEMO_DIR = "demo_data"
shutil.rmtree(DEMO_DIR, ignore_errors=True)

store = RecordStore(DEMO_DIR)
net = Connectivity(online=True)


def show(title):
    print(f"\n--- {title} ---  [{'🟢 ONLINE' if net.is_online() else '🔴 OFFLINE'}]")
    for r in store.list_by_state(*State):
        print(f"  {r['id'][:8]}  {r['state'].value}")


def fake_ai(image_bytes, record_id):
    time.sleep(0.2)  # pretend the AI takes a moment
    return {"age": {"value": 31, "status": "KNOWN", "confidence": 0.93},
            "cin": {"value": None, "status": "NOT_PROVIDED", "confidence": 1.0}}


print("1) Midwife goes offline and photographs 3 pages")
net.set_online(False)
for i in range(3):
    rid = store.capture(f"photo-{i}".encode(), midwife_id="mw-001")
    print(f"  📸 captured {rid[:8]}  💾 saved locally  ⏳ waiting for connection")
process_queue(store, net, fake_ai)
show("Still offline: everything waits in the queue")

print("\n2) Connection comes back, but dies again during the 2nd photo")
net.set_online(True)
calls = {"n": 0}


def flaky_ai(image_bytes, record_id):
    calls["n"] += 1
    if calls["n"] == 2:
        net.set_online(False)
        raise ConnectionError("connection lost")
    return fake_ai(image_bytes, record_id)


process_queue(store, net, flaky_ai)
show("1 processed, 2 still waiting (nothing lost)")

print("\n3) Connection restored for good")
net.set_online(True)
process_queue(store, net, fake_ai)
show("All processed and ready for the midwife to review")
