#ifndef BRAIN_H
#define BRAIN_H
#include <stddef.h>

/* Genome layout (must match orchestrator/genome.py):
 *   W1[H*P], b1[H], W2[A*H], b2[A]      all float32, little-endian
 * Forward:  h = tanh(W1 x + b1) ;  y = tanh(W2 h + b2)
 * y[0..2] = thrust vector, y[3] = reproduce intent.
 */
typedef struct {
    int P, H, A;
    float *W1, *b1, *W2, *b2;  /* point into `weights` */
    float *weights;            /* owned copy of the genome */
    float *h;                  /* hidden scratch */
} Brain;

int  brain_init(Brain *b, int P, int H, int A,
                const float *genome, size_t genome_len);
void brain_step(const Brain *b, const float *x, float *y);
void brain_free(Brain *b);

#endif
