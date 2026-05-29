# ALIFE — 3D Distributed Artificial-Life Simulator (Proof of Concept)

A runnable first slice of the three-part design: a **Python orchestrator** that
owns the world and a supervision web UI, a **world simulator** over GPU-style
data buffers, and **artificial-life agents written in C** that connect over
sockets and can run on other machines.

The headline idea from the design is implemented here: each organism's **body**
(position, velocity, energy — lives in the world) is split from its **brain**
(perception → action controller — lives in a remote C process). Every tick the
world produces a perception vector per agent, ships it to the owning host, the
host's neural brain returns an action vector, and the world applies it.

This PoC makes the world **3D** and visualises it live in the browser with
**WebGL (Three.js)**.

---

## What you'll see

Open the UI and you get a dark "instrument" console around a 3D box world:
little spheres (agents, coloured blue→yellow by energy) swimming toward glowing
food particles, eating, gaining energy, and reproducing (with mutation) when
they have enough. Orbit/zoom with the mouse. The panel shows live tick rate,
population, food, host count, and **sliders that retune the world while it runs**
(metabolism, food rate, mutation, thrust, reproduction energy, …). Pause/speed
controls are there too.

The first generation is seeded with a hand-biased "forager" brain so there is
visible purposeful behaviour immediately; mutation then diversifies it.

---

## Run it

Requirements: Python 3.10+, a C compiler (`cc`/`gcc`), and these Python packages:

```bash
pip install numpy fastapi "uvicorn[standard]"
./run.sh
# then open http://localhost:8000
```

`run.sh` builds the C agent host, starts the orchestrator (web UI on `:8000`,
agent socket on `:9000`), and launches one local agent host.

### Multiple machines / multiple hosts

The agents are designed to run anywhere. Start additional hosts pointed at the
orchestrator's IP — on the same box or across the network:

```bash
./agent_host/agent_host <orchestrator-ip> 9000 host-2
```

New births are assigned to the **least-loaded** host. If a host disconnects, its
organisms are **orphaned and migrated** to the remaining hosts automatically
(their genomes are re-sent), so the simulation survives host failure.

---

## Architecture

```
                ┌──────────────────────── Orchestrator (Python, asyncio) ─────────────────────┐
                │                                                                              │
 Browser ◀──WS/HTTP──▶  FastAPI supervision server  ──┐                                        │
 (Three.js WebGL)        :8000  (state stream, config) │                                       │
                │                                       ▼                                       │
                │   fixed-rate tick loop:  sense ▶ dispatch ▶ (apply latest action) ▶ step     │
                │                              │                         ▲                      │
                │            World (SoA buffers, 3D physics, food) ──────┘                      │
                │            genomes + lifecycle (birth/death/mutation)                         │
                └───────────────────────────────┬──────────────────────────────────────────────┘
                                                 │  TCP :9000  (length-prefixed binary frames)
                            ┌────────────────────┼────────────────────┐
                            ▼                     ▼                    ▼
                     Agent host (C)        Agent host (C)        Agent host (C)
                     many MLP brains       many MLP brains       …  (other machines)
```

**Data plane** (high frequency, fixed shape): `PERCEPTION` and `ACTION` batches
are packed binary (`body_id` + float vector per record). **Control plane**
(`HELLO`/`WELCOME`/`ASSIGN`/`RELEASE`): also binary here, see below.

**Body/brain split.** Bodies are rows in the world's Structure-of-Arrays
buffers. Brains are small MLPs whose weights *are* the genome
(`W1·x → tanh → W2·h → tanh`), instantiated inside the C host on `ASSIGN`. The
orchestrator owns genomes and performs reproduction/mutation centrally, then
ships the child genome to a host.

**Synchronisation.** The tick loop never blocks on the network: it applies the
**latest action received** per agent (the design's bounded-reaction-latency
model). Locally that's same-tick; under real latency an action computed at tick
T simply lands a few ticks later.

### Perception (P=10) and Action (A=4)

```
perception = [ energy, speed,
               food_dir.x, food_dir.y, food_dir.z, food_dist,
               nbr_dir.x,  nbr_dir.y,  nbr_dir.z,  nbr_dist ]
action     = [ thrust.x, thrust.y, thrust.z, reproduce_intent ]
```

---

## What is real vs. stubbed (and why)

This is a **proof of concept**, so some pieces of the full design are stood in
with simpler equivalents chosen to keep it dependency-free and runnable. Every
substitution preserves the *interfaces* so the real component drops in later.

| Design component        | Full design                | In this PoC                                   | Drop-in path |
|-------------------------|----------------------------|-----------------------------------------------|--------------|
| World physics           | OpenCL kernels on GPU      | NumPy on the **same SoA buffer layout**       | `kernels.cl` is the ready Phase-2 port; swap the backend in `world.py` |
| Control-plane encoding  | MessagePack                | Packed binary (so the C host needs no deps)   | Add msgpack to `protocol.py` + host |
| Resources               | 3D scalar resource field   | Discrete food particles                       | Replace food arrays with a voxel grid kernel |
| Reaction latency        | Tunable L-tick delay buffer| "Apply latest received" (non-blocking)        | Buffer perceptions by tick |
| Evolution               | Genome ops on host or orch | Centralised mutate-on-reproduce in orchestrator | Already modular in `genome.py` |

The distributed parts are **real**: real TCP sockets, real C agent processes,
real multi-host assignment, real migration on disconnect.

---

## File tree

```
alife_poc/
├── run.sh                     launcher (build host + start orchestrator + 1 host)
├── kernels.cl                 reference OpenCL kernels (Phase-2 GPU drop-in)
├── orchestrator/
│   ├── server.py              asyncio tick loop + agent TCP server + FastAPI/WS web server
│   ├── world.py               3D world: NumPy SoA physics, food, sensing  (GPU stand-in)
│   ├── genome.py              MLP genome: seed forager + mutation
│   ├── protocol.py            wire framing + binary message codecs
│   ├── config.py              live-tunable parameters (drive the UI sliders)
│   └── web/
│       └── index.html         Three.js / WebGL 3D supervision UI
└── agent_host/
    ├── agent_host.c           one process, many brains, batched perception/action over TCP
    ├── brain.h / brain.c      tiny MLP brain (genome = weights)
    └── Makefile
```

---

## Verified behaviour

The PoC was run end-to-end:

* perception→action round trip across the Python↔C socket;
* population grows from foraging→eating→reproduction (e.g. 120 → ~190 in 5 s);
* live config changes take effect mid-run;
* with two hosts, new births balance to the least-loaded host;
* killing a host migrates its organisms to the survivor (no population loss).

---

## Sensible next steps

1. **Wire PyOpenCL** using `kernels.cl` — the buffer layout already matches.
2. **MessagePack control plane** + a proper `HEARTBEAT`/health timeout.
3. **Bounded-latency buffer** with a configurable L and the lag safety-valve
   from the design (throttle tick rate if a host falls > K ticks behind).
4. **Richer brains / real evolution**: larger nets, recurrent state, selection
   pressure and lineage tracking; expose fitness in the UI.
5. **Spatial hashing** in `sense` (uniform grid) to scale past O(N²).
6. **Binary WebSocket** state frames + instanced LOD to push the UI to 10k+ agents.
```
