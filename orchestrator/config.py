"""Tunable parameters. The web UI edits these live; world.py reads them each tick.

In the full design this is the ConfigManager that also version-stamps changes and
pushes deltas to the GPU kernel args and to agent hosts.  Here it is a plain dict
with bounds so the UI can render sliders.
"""

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
WORLD_SIZE = 200.0      # cube edge; world spans [0, WORLD_SIZE]^3
CAPACITY = 1100         # max simultaneous bodies (array size)
FOOD_CAP = 600
P = 10                  # perception vector length
A = 4                   # action vector length  (thrust xyz + reproduce intent)
H = 12                  # brain hidden units
UI_HZ = 15.0
INITIAL_POP = 120
INITIAL_FOOD = 200


class Config:
    def __init__(self):
        self.v = {k: float(s[0]) for k, s in SPEC.items()}

    def __getattr__(self, k):
        # allow config.thrust_scale style access
        if k == "v":
            raise AttributeError(k)
        return self.v[k]

    def set(self, key, value):
        if key not in SPEC:
            raise KeyError(key)
        _, lo, hi = SPEC[key]
        self.v[key] = float(min(max(value, lo), hi))

    def as_ui(self):
        return {k: {"value": self.v[k], "min": s[1], "max": s[2]}
                for k, s in SPEC.items()}
