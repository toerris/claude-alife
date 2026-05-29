"""Orchestrator / supervisor.

Single asyncio event loop drives three things at once:
  * the fixed-rate tick loop (sense -> dispatch -> apply -> step)
  * a raw TCP server that agent hosts connect to (the data + control plane)
  * a FastAPI + WebSocket supervision server for the browser UI

Body/brain split:  bodies live in world.py, brains live in the connected C hosts.
Genomes are owned here; reproduction mutates the parent genome and ASSIGNs the
child to the least-loaded host.  Actions are applied non-blocking ("use the
latest action received"), which is the reaction-latency model from the design
with the GPU never stalling on the network.
"""
import asyncio
import json
import os
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.requests import Request

import protocol as proto
import config as C
from config import Config, P, A, H
from world import World
import genome as gn

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
AGENT_PORT = 9000
WEB_PORT = 8000


class Host:
    def __init__(self, hid, name, writer):
        self.id = hid
        self.name = name
        self.writer = writer
        self.assigned = set()       # body_ids whose brain lives on this host
        self.last_tick = 0

    def send(self, data: bytes):
        try:
            self.writer.write(data)
        except Exception:
            pass


class Orchestrator:
    def __init__(self):
        self.cfg = Config()
        self.rng = np.random.default_rng(7)
        self.world = World(self.cfg, C.WORLD_SIZE, C.CAPACITY, C.FOOD_CAP, self.rng)
        self.genomes = {}            # body_id -> np.float32 genome
        self.hosts = {}              # host_id -> Host
        self._next_hid = 1
        self.latest_action = {}      # body_id -> np.ndarray(A)
        self.unassigned = set()      # alive bodies with no host yet
        self.paused = False
        self.speed = 1.0
        self.ws_clients = set()
        self.sim_fps = 0.0
        self._seed_world()

    # ---- setup ----
    def _seed_world(self):
        self.world.spawn_food(C.INITIAL_FOOD)
        for _ in range(C.INITIAL_POP):
            pos = self.rng.uniform(0, C.WORLD_SIZE, 3).astype(np.float32)
            bid = self.world.spawn(pos, self.cfg.repro_threshold * 0.6)
            self.genomes[bid] = gn.seed_forager(self.rng)
            self.unassigned.add(bid)

    # ---- host bookkeeping ----
    def _least_loaded(self):
        if not self.hosts:
            return None
        return min(self.hosts.values(), key=lambda h: len(h.assigned))

    def assign_pending(self):
        """Assign any unassigned alive bodies to the least-loaded host."""
        if not self.hosts or not self.unassigned:
            return
        by_host = {}
        for bid in list(self.unassigned):
            if bid not in self.world.id2slot:        # died before assignment
                self.unassigned.discard(bid)
                continue
            h = self._least_loaded()
            h.assigned.add(bid)
            by_host.setdefault(h, []).append((bid, self.genomes[bid]))
            self.unassigned.discard(bid)
        for h, items in by_host.items():
            h.send(proto.enc_assign(items))

    def release_bodies(self, body_ids):
        per_host = {}
        for bid in body_ids:
            for h in self.hosts.values():
                if bid in h.assigned:
                    h.assigned.discard(bid)
                    per_host.setdefault(h, []).append(bid)
            self.unassigned.discard(bid)
            self.latest_action.pop(bid, None)
        for h, ids in per_host.items():
            h.send(proto.enc_release(ids))

    def host_disconnected(self, host: Host):
        # orphan this host's bodies -> reassigned (genome re-sent) next tick.
        # demonstrates brain migration / fault tolerance.
        self.hosts.pop(host.id, None)
        for bid in host.assigned:
            if bid in self.world.id2slot:
                self.unassigned.add(bid)
        print(f"[orch] host {host.id} ({host.name}) disconnected; "
              f"{len(host.assigned)} bodies orphaned -> will migrate")

    # ---- per-tick dispatch ----
    def dispatch(self, tick, ids, perc):
        if len(ids) == 0:
            return
        row = {int(i): perc[k] for k, i in enumerate(ids)}
        for h in self.hosts.values():
            mine = [b for b in h.assigned if b in row]
            if not mine:
                continue
            bid_arr = np.asarray(mine, np.uint32)
            pmat = np.stack([row[b] for b in mine]).astype(np.float32)
            h.send(proto.enc_perception(tick, bid_arr, pmat))

    def on_action_batch(self, host: Host, tick, ids, acts):
        host.last_tick = tick
        for k, bid in enumerate(ids):
            self.latest_action[int(bid)] = acts[k]

    # ---- lifecycle from world.step ----
    def handle_events(self, deaths, repro):
        if deaths:
            self.release_bodies(deaths)
            for bid in deaths:
                self.world.kill(bid)
                self.genomes.pop(bid, None)
        for parent in repro:
            if self.world.pop >= self.cfg.max_population:
                break
            if parent not in self.world.id2slot:
                continue
            pe = self.world.energy_of(parent)
            give = pe * self.cfg.repro_fraction
            ppos = self.world.pos_of(parent)
            cpos = np.clip(ppos + self.rng.normal(0, 4, 3), 0, C.WORLD_SIZE).astype(np.float32)
            child = self.world.spawn(cpos, give)
            if child is None:
                continue
            self.world.add_energy(parent, -give)
            self.genomes[child] = gn.mutate(self.genomes[parent], self.cfg.mutation_rate, self.rng)
            self.unassigned.add(child)

    # ---- the tick loop ----
    async def run_sim(self):
        import time
        t_prev = time.perf_counter()
        ema = None
        ui_every = max(1, int(self.cfg.tick_hz / C.UI_HZ))
        while True:
            dt_target = 1.0 / self.cfg.tick_hz
            if not self.paused:
                # apply latest actions (reaction-latency model, non-blocking)
                if self.latest_action:
                    self.world.apply_actions(self.latest_action)
                deaths, repro = self.world.step(0.05)
                self.handle_events(deaths, repro)
                self.assign_pending()
                ids, perc = self.world.sense()
                self.dispatch(self.world.tick, ids, perc)
                if self.world.tick % ui_every == 0:
                    await self.broadcast_state()
                now = time.perf_counter()
                inst = 1.0 / max(now - t_prev, 1e-6)
                ema = inst if ema is None else 0.9 * ema + 0.1 * inst
                self.sim_fps = ema
                t_prev = now
            else:
                t_prev = time.perf_counter()
            await asyncio.sleep(dt_target / max(self.speed, 0.05))

    # ---- TCP server for agent hosts ----
    async def handle_host(self, reader, writer):
        peer = writer.get_extra_info("peername")
        host = None
        try:
            payload = await proto.read_frame(reader)
            if not payload or payload[0] != proto.HELLO:
                writer.close(); return
            name, cap = proto.dec_hello(payload)
            hid = self._next_hid; self._next_hid += 1
            host = Host(hid, name, writer)
            self.hosts[hid] = host
            writer.write(proto.enc_welcome(hid, P, A, H))
            await writer.drain()
            print(f"[orch] host {hid} '{name}' connected from {peer} (cap {cap})")
            while True:
                payload = await proto.read_frame(reader)
                if payload[0] == proto.ACTION:
                    tick, ids, acts = proto.dec_action(payload, A)
                    self.on_action_batch(host, tick, ids, acts)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if host:
                self.host_disconnected(host)
            try:
                writer.close()
            except Exception:
                pass

    async def broadcast_state(self):
        if not self.ws_clients:
            return
        state = self.world.view()
        state["sim_fps"] = round(self.sim_fps, 1)
        state["paused"] = self.paused
        state["speed"] = self.speed
        state["hosts"] = [{"id": h.id, "name": h.name, "n": len(h.assigned)}
                          for h in self.hosts.values()]
        msg = json.dumps(state)
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.discard(ws)


orch = Orchestrator()
app = FastAPI()


@app.get("/")
async def index():
    with open(os.path.join(WEB_DIR, "index.html")) as f:
        return HTMLResponse(f.read())


@app.get("/api/config")
async def get_config():
    return JSONResponse(orch.cfg.as_ui())


@app.post("/api/config")
async def set_config(req: Request):
    data = await req.json()
    for k, v in data.items():
        try:
            orch.cfg.set(k, float(v))
        except (KeyError, ValueError):
            pass
    return JSONResponse({"ok": True, "config": orch.cfg.as_ui()})


@app.post("/api/control")
async def control(req: Request):
    data = await req.json()
    if "paused" in data:
        orch.paused = bool(data["paused"])
    if "speed" in data:
        orch.speed = float(data["speed"])
    return JSONResponse({"paused": orch.paused, "speed": orch.speed})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    orch.ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()   # we don't expect inbound; keepalive
    except WebSocketDisconnect:
        pass
    finally:
        orch.ws_clients.discard(ws)


async def main():
    tcp = await asyncio.start_server(orch.handle_host, "0.0.0.0", AGENT_PORT)
    print(f"[orch] agent TCP server on :{AGENT_PORT}")
    print(f"[orch] web UI on http://localhost:{WEB_PORT}")
    ucfg = uvicorn.Config(app, host="0.0.0.0", port=WEB_PORT, log_level="warning")
    userver = uvicorn.Server(ucfg)
    await asyncio.gather(tcp.serve_forever(), orch.run_sim(), userver.serve())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
