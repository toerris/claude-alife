/* Agent host: connects to the orchestrator over TCP, runs many brains, and
 * answers each PERCEPTION_BATCH with an ACTION_BATCH.  One process holds many
 * lightweight brains multiplexed over a single socket -- the scalable model
 * from the design (never one process per organism).
 *
 * Dependency-free (POSIX sockets + libc + libm).  Assumes a little-endian host
 * (x86 / ARM) so the wire structs map directly; that matches the orchestrator.
 *
 * Wire: [u32 len][u8 type][body], little-endian.  See orchestrator/protocol.py.
 *   HELLO=1 WELCOME=2 ASSIGN=4 RELEASE=5 PERCEPTION=16 ACTION=17
 */
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include "brain.h"

enum { HELLO=1, WELCOME=2, ASSIGN=4, RELEASE=5, PERCEPTION=16, ACTION=17 };

static int g_P, g_A, g_H;

/* ---------- tiny open-addressing hash map: body_id -> Brain ---------- */
#define MAP_CAP (1u << 15)          /* 32768 slots; PoC pop stays well under */
typedef struct { unsigned id; int state; Brain brain; } Entry; /* state 0 empty,1 used,2 tomb */
static Entry g_map[MAP_CAP];
static unsigned g_count = 0;

static inline unsigned hsh(unsigned k){ return (k * 2654435761u) & (MAP_CAP - 1); }

static Brain *map_put(unsigned id) {
    unsigned i = hsh(id), first_tomb = MAP_CAP;
    for (unsigned n = 0; n < MAP_CAP; n++, i = (i + 1) & (MAP_CAP - 1)) {
        if (g_map[i].state == 1 && g_map[i].id == id) return &g_map[i].brain;
        if (g_map[i].state == 2 && first_tomb == MAP_CAP) first_tomb = i;
        if (g_map[i].state == 0) {
            unsigned dst = (first_tomb != MAP_CAP) ? first_tomb : i;
            g_map[dst].state = 1; g_map[dst].id = id; g_count++;
            return &g_map[dst].brain;
        }
    }
    return NULL; /* full */
}
static Brain *map_get(unsigned id) {
    unsigned i = hsh(id);
    for (unsigned n = 0; n < MAP_CAP; n++, i = (i + 1) & (MAP_CAP - 1)) {
        if (g_map[i].state == 0) return NULL;
        if (g_map[i].state == 1 && g_map[i].id == id) return &g_map[i].brain;
    }
    return NULL;
}
static void map_del(unsigned id) {
    unsigned i = hsh(id);
    for (unsigned n = 0; n < MAP_CAP; n++, i = (i + 1) & (MAP_CAP - 1)) {
        if (g_map[i].state == 0) return;
        if (g_map[i].state == 1 && g_map[i].id == id) {
            brain_free(&g_map[i].brain);
            g_map[i].state = 2; g_count--; return;
        }
    }
}

/* ---------- socket helpers ---------- */
static int read_n(int fd, void *buf, size_t n) {
    char *p = buf; size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, p + got, n - got, 0);
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}
static int write_all(int fd, const void *buf, size_t n) {
    const char *p = buf; size_t sent = 0;
    while (sent < n) {
        ssize_t w = send(fd, p + sent, n - sent, 0);
        if (w <= 0) return -1;
        sent += (size_t)w;
    }
    return 0;
}
/* read one frame body into *out (malloc'd); returns length or -1 */
static long read_frame(int fd, unsigned char **out) {
    unsigned char hdr[4];
    if (read_n(fd, hdr, 4) < 0) return -1;
    unsigned len; memcpy(&len, hdr, 4);          /* LE host */
    unsigned char *buf = malloc(len ? len : 1);
    if (!buf) return -1;
    if (read_n(fd, buf, len) < 0) { free(buf); return -1; }
    *out = buf; return (long)len;
}

static inline unsigned rd_u32(const unsigned char *p){ unsigned v; memcpy(&v,p,4); return v; }
static inline float    rd_f32(const unsigned char *p){ float v; memcpy(&v,p,4); return v; }

/* ---------- message handlers ---------- */
static void handle_assign(const unsigned char *b, long len) {
    (void)len;
    unsigned count = rd_u32(b + 1);
    const unsigned char *p = b + 5;
    for (unsigned k = 0; k < count; k++) {
        unsigned id = rd_u32(p); p += 4;
        unsigned glen = rd_u32(p); p += 4;
        Brain *br = map_put(id);
        if (br) {
            if (brain_init(br, g_P, g_H, g_A, (const float *)p, glen) != 0)
                fprintf(stderr, "[host] brain_init failed for %u\n", id);
        }
        p += (size_t)glen * 4;
    }
}

static void handle_release(const unsigned char *b, long len) {
    (void)len;
    unsigned count = rd_u32(b + 1);
    const unsigned char *p = b + 5;
    for (unsigned k = 0; k < count; k++) { map_del(rd_u32(p)); p += 4; }
}

/* PERCEPTION -> compute -> ACTION (one batched reply) */
static int handle_perception(int fd, const unsigned char *b, long len) {
    (void)len;
    unsigned tick  = rd_u32(b + 1);
    unsigned count = rd_u32(b + 5);
    const unsigned char *p = b + 9;
    const size_t prec = 4 + (size_t)g_P * 4;     /* perception record size */
    const size_t arec = 4 + (size_t)g_A * 4;     /* action record size */

    size_t outlen = 1 + 4 + 4 + (size_t)count * arec;
    unsigned char *out = malloc(4 + outlen);
    unsigned char *w = out + 4;
    unsigned olen = (unsigned)outlen; memcpy(out, &olen, 4);
    *w++ = ACTION;
    memcpy(w, &tick, 4);  w += 4;
    memcpy(w, &count, 4); w += 4;

    float x[64], y[16];
    for (unsigned k = 0; k < count; k++) {
        unsigned id = rd_u32(p);
        for (int j = 0; j < g_P; j++) x[j] = rd_f32(p + 4 + (size_t)j * 4);
        p += prec;
        Brain *br = map_get(id);
        if (br) brain_step(br, x, y);
        else    for (int j = 0; j < g_A; j++) y[j] = 0.0f;  /* no brain -> idle */
        memcpy(w, &id, 4); w += 4;
        memcpy(w, y, (size_t)g_A * 4); w += (size_t)g_A * 4;
    }
    int rc = write_all(fd, out, 4 + outlen);
    free(out);
    return rc;
}

int main(int argc, char **argv) {
    const char *host = argc > 1 ? argv[1] : "127.0.0.1";
    const char *port = argc > 2 ? argv[2] : "9000";
    const char *name = argc > 3 ? argv[3] : "host-c";

    struct addrinfo hints = {0}, *res;
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) { perror("getaddrinfo"); return 1; }
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) { perror("connect"); return 1; }
    freeaddrinfo(res);
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    /* HELLO: type, u16 name_len, name, u32 capacity */
    unsigned short nl = (unsigned short)strlen(name);
    unsigned cap = 100000;
    unsigned hlen = 1 + 2 + nl + 4;
    unsigned char *hb = malloc(4 + hlen), *q = hb + 4;
    memcpy(hb, &hlen, 4);
    *q++ = HELLO; memcpy(q, &nl, 2); q += 2; memcpy(q, name, nl); q += nl; memcpy(q, &cap, 4);
    write_all(fd, hb, 4 + hlen); free(hb);

    /* WELCOME: host_id,P,A,H */
    unsigned char *wb; long wlen = read_frame(fd, &wb);
    if (wlen < 0 || wb[0] != WELCOME) { fprintf(stderr, "no welcome\n"); return 1; }
    unsigned hid = rd_u32(wb + 1);
    g_P = (int)rd_u32(wb + 5); g_A = (int)rd_u32(wb + 9); g_H = (int)rd_u32(wb + 13);
    free(wb);
    printf("[host] connected as id %u; P=%d A=%d H=%d\n", hid, g_P, g_A, g_H);
    fflush(stdout);

    for (;;) {
        unsigned char *b; long len = read_frame(fd, &b);
        if (len < 0) { printf("[host] disconnected\n"); break; }
        switch (b[0]) {
            case ASSIGN:     handle_assign(b, len); break;
            case RELEASE:    handle_release(b, len); break;
            case PERCEPTION: if (handle_perception(fd, b, len) < 0) { free(b); goto done; } break;
            default: break;
        }
        free(b);
    }
done:
    close(fd);
    return 0;
}
