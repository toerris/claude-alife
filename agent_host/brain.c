#include "brain.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int brain_init(Brain *b, int P, int H, int A,
               const float *genome, size_t genome_len)
{
    size_t need = (size_t)H * P + H + (size_t)A * H + A;
    if (genome_len < need) return -1;
    b->P = P; b->H = H; b->A = A;
    b->weights = (float *)malloc(need * sizeof(float));
    b->h = (float *)malloc((size_t)H * sizeof(float));
    if (!b->weights || !b->h) { free(b->weights); free(b->h); return -1; }
    memcpy(b->weights, genome, need * sizeof(float));
    float *w = b->weights;
    b->W1 = w;            w += (size_t)H * P;
    b->b1 = w;            w += H;
    b->W2 = w;            w += (size_t)A * H;
    b->b2 = w;
    return 0;
}

void brain_step(const Brain *b, const float *x, float *y)
{
    int P = b->P, H = b->H, A = b->A;
    for (int i = 0; i < H; i++) {
        float s = b->b1[i];
        const float *row = b->W1 + (size_t)i * P;
        for (int j = 0; j < P; j++) s += row[j] * x[j];
        b->h[i] = tanhf(s);
    }
    for (int o = 0; o < A; o++) {
        float s = b->b2[o];
        const float *row = b->W2 + (size_t)o * H;
        for (int i = 0; i < H; i++) s += row[i] * b->h[i];
        y[o] = tanhf(s);
    }
}

void brain_free(Brain *b)
{
    free(b->weights); free(b->h);
    b->weights = NULL; b->h = NULL;
}
