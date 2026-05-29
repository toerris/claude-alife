"""3D world simulator.

This is the CPU/NumPy stand-in for the OpenCL world.  It deliberately uses the
same Structure-of-Arrays layout the GPU version would use (one contiguous array
per attribute) so the OpenCL kernels in ../kernels.cl can be wired in later with
no change to the orchestrator.  Bodies live here; brains live in the C hosts.

Coordinate system: a cube [0, size]^3.  Food are discrete particles (a 3D field
is the Phase-2 GPU version).
"""
import numpy as np
from config import P, A


class World:
    def __init__(self, cfg, size, capacity, food_cap, rng=None):
        self.cfg = cfg
        self.size = float(size)
        self.cap = capacity
        self.rng = rng or np.random.default_rng(1)

        # --- body state (SoA) ---
        self.pos = np.zeros((capacity, 3), np.float32)
        self.vel = np.zeros((capacity, 3), np.float32)
        self.energy = np.zeros(capacity, np.float32)
        self.age = np.zeros(capacity, np.float32)
        self.alive = np.zeros(capacity, bool)
        self.last_action = np.zeros((capacity, A), np.float32)

        self._free = list(range(capacity - 1, -1, -1))  # stack of free slots
        self.id2slot = {}
        self.slot2id = np.full(capacity, -1, np.int64)
        self._next_id = 1

        # --- food field (discrete particles) ---
        self.food_cap = food_cap
        self.food_pos = np.zeros((food_cap, 3), np.float32)
        self.food_alive = np.zeros(food_cap, bool)

        self.tick = 0

    # ---- population accessors ----
    @property
    def pop(self):
        return len(self.id2slot)

    def alive_ids(self):
        return np.fromiter(self.id2slot.keys(), dtype=np.uint32, count=self.pop)

    def energy_of(self, bid):
        return float(self.energy[self.id2slot[bid]])

    def pos_of(self, bid):
        return self.pos[self.id2slot[bid]].copy()

    # ---- lifecycle ----
    def spawn(self, pos, energy):
        if not self._free:
            return None
        s = self._free.pop()
        bid = self._next_id
        self._next_id += 1
        self.pos[s] = pos
        self.vel[s] = 0.0
        self.energy[s] = energy
        self.age[s] = 0.0
        self.last_action[s] = 0.0
        self.alive[s] = True
        self.id2slot[bid] = s
        self.slot2id[s] = bid
        return bid

    def kill(self, bid):
        s = self.id2slot.pop(bid, None)
        if s is None:
            return
        self.alive[s] = False
        self.slot2id[s] = -1
        self._free.append(s)

    def add_energy(self, bid, de):
        self.energy[self.id2slot[bid]] += de

    # ---- food ----
    def spawn_food(self, n):
        free = np.where(~self.food_alive)[0]
        n = min(n, len(free), int(self.cfg.food_max) - int(self.food_alive.sum()))
        if n <= 0:
            return
        idx = free[:n]
        self.food_pos[idx] = self.rng.uniform(0, self.size, size=(n, 3)).astype(np.float32)
        self.food_alive[idx] = True

    # ---- SENSING (mirrors kernels.cl `sense`) ----
    def sense(self):
        """Returns (ids uint32, perception float32 (n,P)) for all alive bodies."""
        if self.pop == 0:
            return np.empty(0, np.uint32), np.empty((0, P), np.float32)
        ids = self.alive_ids()
        slots = np.array([self.id2slot[int(i)] for i in ids])
        ap = self.pos[slots]                     # (n,3)
        av = self.vel[slots]
        ae = self.energy[slots]
        n = len(ids)
        rng = self.cfg.sensor_range

        perc = np.zeros((n, P), np.float32)
        perc[:, 0] = np.clip(ae / self.cfg.repro_threshold, 0, 2)
        perc[:, 1] = np.linalg.norm(av, axis=1) / self.cfg.max_speed

        # nearest food
        fidx = np.where(self.food_alive)[0]
        if len(fidx):
            fp = self.food_pos[fidx]             # (f,3)
            d = ap[:, None, :] - fp[None, :, :]  # (n,f,3)
            dist2 = np.einsum("nfc,nfc->nf", d, d)
            j = dist2.argmin(axis=1)
            nd = np.sqrt(dist2[np.arange(n), j])
            vec = fp[j] - ap                      # toward food
            norm = np.maximum(nd[:, None], 1e-5)
            perc[:, 2:5] = (vec / norm).astype(np.float32)
            perc[:, 5] = np.clip(nd / rng, 0, 1)
        else:
            perc[:, 5] = 1.0

        # nearest neighbour (other agent)
        if n > 1:
            dd = ap[:, None, :] - ap[None, :, :]
            nd2 = np.einsum("nmc,nmc->nm", dd, dd)
            np.fill_diagonal(nd2, np.inf)
            k = nd2.argmin(axis=1)
            ndist = np.sqrt(nd2[np.arange(n), k])
            nvec = ap[k] - ap
            norm = np.maximum(ndist[:, None], 1e-5)
            perc[:, 6:9] = (nvec / norm).astype(np.float32)
            perc[:, 9] = np.clip(ndist / rng, 0, 1)
        else:
            perc[:, 9] = 1.0

        return ids, perc

    def apply_actions(self, actions):
        """actions: dict body_id -> ndarray (A,) float32."""
        for bid, a in actions.items():
            s = self.id2slot.get(int(bid))
            if s is not None:
                self.last_action[s] = a

    # ---- STEP (mirrors kernels.cl `act` + `integrate` + `environment`) ----
    def step(self, dt):
        self.tick += 1
        cfg = self.cfg
        if self.pop:
            ids = self.alive_ids()
            slots = np.array([self.id2slot[int(i)] for i in ids])
            act = self.last_action[slots]
            thrust = act[:, 0:3] * cfg.thrust_scale

            v = self.vel[slots] + thrust * dt
            v *= max(0.0, 1.0 - cfg.drag * dt)
            sp = np.linalg.norm(v, axis=1, keepdims=True)
            over = (sp > cfg.max_speed).ravel()
            v[over] *= (cfg.max_speed / sp[over])
            p = self.pos[slots] + v * dt

            # reflect off the cube walls
            for c in range(3):
                lo = p[:, c] < 0
                hi = p[:, c] > self.size
                v[lo, c] *= -0.5
                v[hi, c] *= -0.5
                p[:, c] = np.clip(p[:, c], 0, self.size)

            self.vel[slots] = v
            self.pos[slots] = p
            self.age[slots] += 1.0

            speed = np.linalg.norm(v, axis=1)
            self.energy[slots] -= (cfg.basal_metabolism + cfg.movement_cost * speed) * dt

            # --- eating: each agent eats nearest food within eat radius ---
            self._eat(slots)

        self.spawn_food(int(cfg.food_spawn))

        # --- deaths & reproduction requests ---
        deaths, repro = [], []
        if self.pop:
            ids = self.alive_ids()
            for bid in ids:
                s = self.id2slot[int(bid)]
                if self.energy[s] <= 0.0 or self.age[s] > cfg.max_age:
                    deaths.append(int(bid))
                elif (self.energy[s] > cfg.repro_threshold
                      and self.last_action[s, 3] > 0.0
                      and self.pop < cfg.max_population):
                    repro.append(int(bid))
        return deaths, repro

    def _eat(self, slots):
        fidx = np.where(self.food_alive)[0]
        if not len(fidx):
            return
        eat_r2 = (self.size * 0.012 + 2.5) ** 2  # small fixed bite radius
        ap = self.pos[slots]
        fp = self.food_pos[fidx]
        d = ap[:, None, :] - fp[None, :, :]
        dist2 = np.einsum("nfc,nfc->nf", d, d)
        # greedily assign: nearest food per agent if in range and still present
        taken = np.zeros(len(fidx), bool)
        order = np.argsort(dist2.min(axis=1))
        for ai in order:
            j = dist2[ai].argmin()
            if not taken[j] and dist2[ai, j] <= eat_r2:
                taken[j] = True
                self.energy[slots[ai]] += self.cfg.food_value
        gone = fidx[taken]
        self.food_alive[gone] = False

    # ---- view for the web UI ----
    def view(self):
        ids = self.alive_ids()
        a = []
        if len(ids):
            slots = np.array([self.id2slot[int(i)] for i in ids])
            pos = self.pos[slots]
            en = self.energy[slots]
            flat = np.empty((len(ids), 4), np.float32)
            flat[:, :3] = pos
            flat[:, 3] = en / max(self.cfg.repro_threshold, 1e-3)
            a = flat.ravel().tolist()
        f = self.food_pos[self.food_alive].ravel().tolist()
        return {"t": self.tick, "pop": int(self.pop),
                "nfood": int(self.food_alive.sum()),
                "size": self.size, "a": a, "f": f}
