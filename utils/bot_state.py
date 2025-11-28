import json
import os
import datetime

STATE_PATH = os.path.join("env", "bot_state.json")


def _default_state():
    return {"paused": False, "updated_at": datetime.datetime.utcnow().isoformat()}


def _ensure_dir():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                data = json.load(f)
                if "paused" in data:
                    return data
        except (json.JSONDecodeError, OSError):
            pass
    return _default_state()


def save_state(state):
    _ensure_dir()
    state["updated_at"] = datetime.datetime.utcnow().isoformat()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    return state


def is_bot_paused():
    state = load_state()
    return bool(state.get("paused", False))


def set_bot_paused(paused: bool):
    state = load_state()
    state["paused"] = bool(paused)
    return save_state(state)


def status_label():
    state = load_state()
    label = "paused" if state.get("paused") else "running"
    ts = state.get("updated_at", "unknown time")
    return f"{label.upper()} (updated {ts} UTC)"
