// Teto de banda e de throughput por tipo numérico numa MI50 (gfx906). Rodar com
// `scripts/teto-mi50.sh`, que confere a margem de VRAM antes.
//
// Banda: leitura pura em uint4 (16 B por lane), grid-stride, 2 GiB (>> 4 MiB de L2). É o
// "100% da banda" contra o qual medir os matvec do decode — 836 GB/s aqui em 2026-09-25,
// não os 1024 nominais.
// Compute: cadeias independentes de FMA/dot sem tocar memória, para comparar fp64, fp32,
// fp16 empacotado (half2), dot2 f16->f32, dot4 int8 e dot8 int4. O dot8 só existe em HIP:
// `GL_EXT_integer_dot_product` (Vulkan) não tem produto de 4 bits.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void ler(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;  // impede eliminar as leituras
}

constexpr int IT = 4096;
constexpr int CH = 8;  // cadeias independentes por lane (esconde a latência da ALU)

__global__ void k_f64(double* o, double a) {
    double x[CH];
    for (int c = 0; c < CH; c++) x[c] = threadIdx.x + c;
    for (int i = 0; i < IT; i++)
        for (int c = 0; c < CH; c++) x[c] = fma(x[c], a, 1.0);
    double s = 0; for (int c = 0; c < CH; c++) s += x[c];
    if (s == 1.2345) o[0] = s;
}
__global__ void k_f32(float* o, float a) {
    float x[CH];
    for (int c = 0; c < CH; c++) x[c] = threadIdx.x + c;
    for (int i = 0; i < IT; i++)
        for (int c = 0; c < CH; c++) x[c] = fmaf(x[c], a, 1.0f);
    float s = 0; for (int c = 0; c < CH; c++) s += x[c];
    if (s == 1.2345f) o[0] = s;
}
__global__ void k_f16x2(float* o, float a) {
    __half2 x[CH], av = __float2half2_rn(a), one = __float2half2_rn(1.0f);
    for (int c = 0; c < CH; c++) x[c] = __float2half2_rn((float)(threadIdx.x + c));
    for (int i = 0; i < IT; i++)
        for (int c = 0; c < CH; c++) x[c] = __hfma2(x[c], av, one);  // v_pk_fma_f16
    float s = 0; for (int c = 0; c < CH; c++) s += __low2float(x[c]) + __high2float(x[c]);
    if (s == 1.2345f) o[0] = s;
}
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
__global__ void k_dot2(float* o, float a) {
    float acc[CH];
    h2 v = {(_Float16)a, (_Float16)a}, w = {(_Float16)1.0f, (_Float16)0.5f};
    for (int c = 0; c < CH; c++) acc[c] = threadIdx.x + c;
    for (int i = 0; i < IT; i++)
        for (int c = 0; c < CH; c++) acc[c] = __builtin_amdgcn_fdot2(v, w, acc[c], false);  // v_dot2_f32_f16
    float s = 0; for (int c = 0; c < CH; c++) s += acc[c];
    if (s == 1.2345f) o[0] = s;
}
__global__ void k_dot4(int* o, int a) {
    int acc[CH];
    for (int c = 0; c < CH; c++) acc[c] = threadIdx.x + c;
    for (int i = 0; i < IT; i++)
        for (int c = 0; c < CH; c++) acc[c] = __builtin_amdgcn_sdot4(a, 0x01020304, acc[c], false);  // v_dot4_i32_i8
    int s = 0; for (int c = 0; c < CH; c++) s += acc[c];
    if (s == 12345) o[0] = s;
}

__global__ void k_dot8(int* o, int a) {
    int acc[CH];
    for (int c = 0; c < CH; c++) acc[c] = threadIdx.x + c;
    for (int i = 0; i < IT; i++)
        for (int c = 0; c < CH; c++) acc[c] = __builtin_amdgcn_sdot8(a, 0x12345671, acc[c], false);  // v_dot8_i32_i4
    int s = 0; for (int c = 0; c < CH; c++) s += acc[c];
    if (s == 12345) o[0] = s;
}

template <class F>
float cronometra(F f, int reps) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    f(); CK(hipDeviceSynchronize());
    std::vector<float> ms;
    for (int r = 0; r < reps; r++) {
        CK(hipEventRecord(a)); f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float t; CK(hipEventElapsedTime(&t, a, b)); ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];  // mediana
}

int main() {
    int nd; CK(hipGetDeviceCount(&nd));
    const size_t BYTES = 2ull << 30;
    for (int d = 0; d < nd; d++) {
        CK(hipSetDevice(d));
        hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, d));
        printf("== device %d %s pci %02x:%02x  CUs=%d  clock=%d MHz\n", d, pr.gcnArchName,
               pr.pciBusID, pr.pciDeviceID, pr.multiProcessorCount, pr.clockRate / 1000);
        uint4* buf; unsigned* out; CK(hipMalloc(&buf, BYTES)); CK(hipMalloc(&out, 64));
        CK(hipMemset(buf, 1, BYTES));
        size_t n = BYTES / 16;
        for (int wpcu : {4, 8, 16, 32}) {
            int grid = pr.multiProcessorCount * wpcu;
            float ms = cronometra([&] { ler<<<grid, 256>>>(buf, n, out); }, 20);
            printf("  leitura 2 GiB, %2d wg/CU: %7.1f GB/s\n", wpcu, BYTES / (ms * 1e6));
        }
        CK(hipFree(buf));
        int grid = pr.multiProcessorCount * 16, blk = 256;
        double ops = (double)grid * blk * IT * CH;  // instruções por lane-op
        float ms;
        ms = cronometra([&] { k_f64<<<grid, blk>>>((double*)out, 1.0000001); }, 5);
        printf("  fp64 fma       : %6.2f TFLOPS\n", ops * 2 / (ms * 1e9));
        ms = cronometra([&] { k_f32<<<grid, blk>>>((float*)out, 1.0000001f); }, 5);
        printf("  fp32 fma       : %6.2f TFLOPS\n", ops * 2 / (ms * 1e9));
        ms = cronometra([&] { k_f16x2<<<grid, blk>>>((float*)out, 0.999f); }, 5);
        printf("  fp16 pk_fma    : %6.2f TFLOPS\n", ops * 4 / (ms * 1e9));
        ms = cronometra([&] { k_dot2<<<grid, blk>>>((float*)out, 0.999f); }, 5);
        printf("  dot2 f16->f32  : %6.2f TFLOPS\n", ops * 4 / (ms * 1e9));
        ms = cronometra([&] { k_dot4<<<grid, blk>>>((int*)out, 0x01010101); }, 5);
        printf("  dot4 int8      : %6.2f TOPS\n", ops * 8 / (ms * 1e9));
        ms = cronometra([&] { k_dot8<<<grid, blk>>>((int*)out, 0x11111111); }, 5);
        printf("  dot8 int4      : %6.2f TOPS\n", ops * 16 / (ms * 1e9));
        CK(hipFree(out));
    }
}
