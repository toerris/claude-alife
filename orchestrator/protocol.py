"""Wire protocol for orchestrator <-> agent-host sockets.

Framing:  [uint32 length][payload]
payload:  [uint8 type][type-specific body]   (everything little-endian)

The full design uses MessagePack for the (flexible, low-frequency) control plane
and packed binary for the (fixed-shape, high-frequency) data plane.  For this PoC
*everything* is packed binary so the C agent host has zero external dependencies.
The C structs in agent_host.c mirror these layouts byte-for-byte.
"""
import struct
import asyncio
import numpy as np

# message types
HELLO = 1        # host -> orch
WELCOME = 2      # orch -> host
ASSIGN = 4       # orch -> host : give bodies + genomes -> instantiate brains
RELEASE = 5      # orch -> host : destroy brains
PERCEPTION = 16  # orch -> host : per-tick perception batch
ACTION = 17      # host -> orch : per-tick action batch

_LEN = struct.Struct("<I")


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one length-prefixed frame; returns the payload (type byte + body)."""
    hdr = await reader.readexactly(4)
    (n,) = _LEN.unpack(hdr)
    return await reader.readexactly(n)


def frame(payload: bytes) -> bytes:
    return _LEN.pack(len(payload)) + payload


# ---- encoders (orchestrator side) ----------------------------------------

def enc_welcome(host_id: int, P: int, A: int, H: int) -> bytes:
    return frame(struct.pack("<BIIII", WELCOME, host_id, P, A, H))


def enc_assign(items) -> bytes:
    """items: list of (body_id:int, genome:np.ndarray float32)."""
    parts = [struct.pack("<BI", ASSIGN, len(items))]
    for bid, genome in items:
        g = np.ascontiguousarray(genome, dtype="<f4")
        parts.append(struct.pack("<II", bid, g.size))
        parts.append(g.tobytes())
    return frame(b"".join(parts))


def enc_release(body_ids) -> bytes:
    body_ids = list(body_ids)
    return frame(struct.pack("<BI", RELEASE, len(body_ids)) +
                 np.asarray(body_ids, dtype="<u4").tobytes())


def enc_perception(tick: int, ids: np.ndarray, perc: np.ndarray) -> bytes:
    """ids: (n,) uint32 ; perc: (n,P) float32. One record = id + P floats."""
    n, P = perc.shape
    rec = np.zeros(n, dtype=[("id", "<u4"), ("p", "<f4", (P,))])
    rec["id"] = ids
    rec["p"] = perc
    return frame(struct.pack("<BII", PERCEPTION, tick, n) + rec.tobytes())


# ---- decoders --------------------------------------------------------------

def dec_hello(body: bytes):
    name_len = struct.unpack_from("<H", body, 1)[0]
    name = body[3:3 + name_len].decode("utf-8", "replace")
    cap = struct.unpack_from("<I", body, 3 + name_len)[0]
    return name, cap


def dec_action(body: bytes, A: int):
    """Returns (tick, ids ndarray uint32, actions ndarray float32 (n,A))."""
    tick, n = struct.unpack_from("<II", body, 1)
    rec = np.frombuffer(body, dtype=[("id", "<u4"), ("a", "<f4", (A,))],
                        count=n, offset=9)
    return tick, rec["id"].copy(), rec["a"].copy()


def enc_hello(name: str, capacity: int) -> bytes:  # used by a python test client
    nb = name.encode()
    return frame(struct.pack("<BH", HELLO, len(nb)) + nb + struct.pack("<I", capacity))
