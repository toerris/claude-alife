"""Tunable parameters. The web UI edits these live; world.py reads them each tick.

In the full design this is the ConfigManager that also version-stamps changes and
pushes deltas to the GPU kernel args and to agent hosts. Here it is a plain dict
with bounds so the UI can render sliders.

The defaults below can be overridden with ALIFE_* environment variables. This
keeps the PoC easy to run locally while borrowing the production-friendly
configuration style from the larger scaffold.
"""
import os

ENV_PREFIX = "ALIFE_"


def _env_name(name):
    return ENV_PREFIX + name.upper()


def _env_float(name, default):
    raw = os.getenv(_env_name(name))
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"[config] ignoring invalid {_env_name(name)}={raw!r}; using {default!r}")
        return float(default)


def _env_int(name, default):
    return int(round(_env_float(name, default)))


def _clamp(value, lo, hi):
    return float(min(max(value, lo), hi))


# value, min, max  (only entries with bounds are exposed as sliders)
SPEC = {
    "thrust_scale":      (6.0,   0.0,  20.0),
    "drag":              (0.9,   0.0,   3.0),
    "max_speed":         (8.0,   1.0,  20.0),
    "basal_metabolism":  (0.8,   0.0,   5.0),
    "movement_cost":     (0.15,  0.0,   1.0),
    "food_value":        (16.0,  1.0,  60.0),
    "food_max":          (350,   0,    600),
    "food_spawn":        (6,     0,     40),   # food spawned per tick (up to food_max)
    "repro_threshold":   (45.0, 10.0, 120.0),
    "repro_fraction":    (0.45,  0.1,   0.9),  # energy fraction handed to child
    "mutation_rate":     (0.08,  0.0,   0.5),
    "sensor_range":      (40.0,  5.0, 120.0),
    "max_age":           (1200,  50,  5000),
    "max_population":    (450,   10,  1000),
    "tick_hz":           (30.0,  1.0,  60.0),
}

# fixed (not exposed as sliders)
WORLD_SIZE = _env_float("world_size", 200.0)      # cube edge; world spans [0, WORLD_SIZE]^3
CAPACITY = _env_int("capacity", 1100)             # max simultaneous bodies (array size)
FOOD_CAP = _env_int("food_cap", 600)
P = _env_int("perception_size", 10)               # perception vector length
A = _env_int("action_size", 4)                    # action vector length (thrust xyz + reproduce intent)
H = _env_int("hidden_units", 12)                  # brain hidden units
UI_HZ = _env_float("ui_hz", 15.0)
INITIAL_POP = _env_int("initial_pop", 120)
INITIAL_FOOD = _env_int("initial_food", 200)

# service/runtime settings
AGENT_HOST = os.getenv(_env_name("agent_host"), "0.0.0.0")
AGENT_PORT = _env_int("agent_port", 9000)
WEB_HOST = os.getenv(_env_name("web_host"), "0.0.0.0")
WEB_PORT = _env_int("web_port", 8000)
SNAPSHOT_DIR = os.getenv(_env_name("snapshot_dir"), "snapshots")


class Config:
    def __init__(self):
        self.v = {}
        for k, (default, lo, hi) in SPEC.items():
            self.v[k] = _clamp(_env_float(k, default), lo, hi)

    def __getattr__(self, k):
        # allow config.thrust_scale style access
        if k == "v":
            raise AttributeError(k)
        return self.v[k]

    def set(self, key, value):
        if key not in SPEC:
            raise KeyError(key)
        _, lo, hi = SPEC[key]
        self.v[key] = _clamp(float(value), lo, hi)

    def as_plain(self):
        return dict(self.v)

    def as_ui(self):
        return {
            k: {"value": self.v[k], "min": s[1], "max": s[2], "env": _env_name(k)}
            for k, s in SPEC.items()
        }


def fixed_settings():
    return {
        "world_size": WORLD_SIZE,
        "capacity": CAPACITY,
        "food_cap": FOOD_CAP,
        "perception_size": P,
        "action_size": A,
        "hidden_units": H,
        "ui_hz": UI_HZ,
        "initial_pop": INITIAL_POP,
        "initial_food": INITIAL_FOOD,
        "agent_host": AGENT_HOST,
        "agent_port": AGENT_PORT,
        "web_host": WEB_HOST,
        "web_port": WEB_PORT,
        "snapshot_dir": SNAPSHOT_DIR,
    }
