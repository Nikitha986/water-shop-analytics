import time
import uuid
from config import VISIT_TIMEOUT

active_tracks = {}

def assign_id(bbox):
    x, y, w, h = bbox
    key = f"{x//50}-{y//50}"

    now = time.time()

    if key in active_tracks:
        if now - active_tracks[key]["time"] < VISIT_TIMEOUT:
            active_tracks[key]["time"] = now
            return active_tracks[key]["id"]

    new_id = str(uuid.uuid4())[:8]
    active_tracks[key] = {"id": new_id, "time": now}
    return new_id
