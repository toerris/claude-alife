/* kernels.cl -- REFERENCE OpenCL kernels for the world simulator.
 *
 * STATUS: not wired into the PoC. The PoC runs the identical logic in NumPy
 * (orchestrator/world.py) so it works without a GPU. These kernels are the
 * Phase-2 drop-in: same Structure-of-Arrays buffers, same perception/action
 * layout, so swapping the NumPy backend for PyOpenCL touches only the host
 * (PyOpenCL setup, buffer alloc, enqueue) -- not the orchestrator or the agents.
 *
 * Buffers (SoA), all device-resident:
 *   pos[N*3], vel[N*3], energy[N], age[N], alive[N], last_action[N*A]
 *   food_pos[F*3], food_alive[F]
 *   perception[N*P]   (written by `sense`, copied to host)
 * P=10 perception, A=4 action. Spatial hashing is omitted for brevity (the
 * PoC's N is small); for scale, add a uniform-grid build kernel before `sense`.
 */

#define P 10
#define A 4

/* SENSE: per agent, pack [energy, speed, food_dir(3), food_dist,
 *                          nbr_dir(3), nbr_dist] into perception[]. */
__kernel void sense(__global const float *pos,
                    __global const float *vel,
                    __global const float *energy,
                    __global const uchar *alive,
                    __global const float *food_pos,
                    __global const uchar *food_alive,
                    __global float *perception,
                    const int N, const int F,
                    const float sensor_range,
                    const float repro_threshold,
                    const float max_speed)
{
    int i = get_global_id(0);
    if (i >= N || !alive[i]) return;
    float3 p = (float3)(pos[i*3], pos[i*3+1], pos[i*3+2]);
    float3 v = (float3)(vel[i*3], vel[i*3+1], vel[i*3+2]);
    __global float *o = perception + i*P;

    o[0] = clamp(energy[i] / repro_threshold, 0.0f, 2.0f);
    o[1] = length(v) / max_speed;

    /* nearest food */
    float best = INFINITY; int bj = -1;
    for (int j = 0; j < F; j++) {
        if (!food_alive[j]) continue;
        float3 fp = (float3)(food_pos[j*3], food_pos[j*3+1], food_pos[j*3+2]);
        float d2 = dot(fp - p, fp - p);
        if (d2 < best) { best = d2; bj = j; }
    }
    if (bj >= 0) {
        float3 fp = (float3)(food_pos[bj*3], food_pos[bj*3+1], food_pos[bj*3+2]);
        float d = sqrt(best); float3 dir = (fp - p) / fmax(d, 1e-5f);
        o[2]=dir.x; o[3]=dir.y; o[4]=dir.z; o[5]=clamp(d/sensor_range,0.0f,1.0f);
    } else { o[2]=o[3]=o[4]=0; o[5]=1; }

    /* nearest neighbour (brute force; replace with grid for scale) */
    best = INFINITY; bj = -1;
    for (int k = 0; k < N; k++) {
        if (k == i || !alive[k]) continue;
        float3 q = (float3)(pos[k*3], pos[k*3+1], pos[k*3+2]);
        float d2 = dot(q - p, q - p);
        if (d2 < best) { best = d2; bj = k; }
    }
    if (bj >= 0) {
        float3 q = (float3)(pos[bj*3], pos[bj*3+1], pos[bj*3+2]);
        float d = sqrt(best); float3 dir = (q - p) / fmax(d, 1e-5f);
        o[6]=dir.x; o[7]=dir.y; o[8]=dir.z; o[9]=clamp(d/sensor_range,0.0f,1.0f);
    } else { o[6]=o[7]=o[8]=0; o[9]=1; }
}

/* INTEGRATE: apply last_action thrust, drag, speed clamp, wall bounce,
 * metabolism. (Eating / births / deaths handled in a follow-up `act` kernel
 * that uses atomics to append to event lists the host drains.) */
__kernel void integrate(__global float *pos,
                        __global float *vel,
                        __global float *energy,
                        __global float *age,
                        __global const uchar *alive,
                        __global const float *last_action,
                        const int N, const float dt, const float size,
                        const float thrust_scale, const float drag,
                        const float max_speed, const float basal,
                        const float move_cost)
{
    int i = get_global_id(0);
    if (i >= N || !alive[i]) return;
    float3 p = (float3)(pos[i*3], pos[i*3+1], pos[i*3+2]);
    float3 v = (float3)(vel[i*3], vel[i*3+1], vel[i*3+2]);
    float3 a = (float3)(last_action[i*A], last_action[i*A+1], last_action[i*A+2]);

    v += a * thrust_scale * dt;
    v *= fmax(0.0f, 1.0f - drag * dt);
    float sp = length(v);
    if (sp > max_speed) v *= (max_speed / sp);
    p += v * dt;

    /* reflect off cube walls */
    if (p.x < 0)    { p.x = 0;    v.x = -0.5f*v.x; }
    if (p.x > size) { p.x = size; v.x = -0.5f*v.x; }
    if (p.y < 0)    { p.y = 0;    v.y = -0.5f*v.y; }
    if (p.y > size) { p.y = size; v.y = -0.5f*v.y; }
    if (p.z < 0)    { p.z = 0;    v.z = -0.5f*v.z; }
    if (p.z > size) { p.z = size; v.z = -0.5f*v.z; }

    pos[i*3]=p.x; pos[i*3+1]=p.y; pos[i*3+2]=p.z;
    vel[i*3]=v.x; vel[i*3+1]=v.y; vel[i*3+2]=v.z;
    age[i] += 1.0f;
    energy[i] -= (basal + move_cost * length(v)) * dt;
}
