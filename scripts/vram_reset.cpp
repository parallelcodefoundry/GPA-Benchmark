// vram_reset.cpp -- GPA-G1 J0 protocol T0 (from agent M's vramtool "fill" mode).
// Allocates all free VRAM on the visible GCD (1 GiB chunks until one fails, then 64 MiB chunks),
// frees everything and exits 0. Run before every timed process so each starts from the same
// VRAM allocator state (M.md Q1/Q4: the bfs/nw "levels" are physical-placement states that this
// reset makes sticky). No hipMemset: only the allocator state matters.
// With one argument G (GiB, e.g. 3.5) it instead allocates ~G GiB in 64 MiB chunks and frees it:
// the "perturb" step of the post-batch multi-GCD rescoring (M.md T6).
// Build: hipcc -O2 -o frontier_tools/vram_reset scripts/vram_reset.cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  std::vector<void*> v;
  size_t used = 0;
  if (argc > 1) {
    const size_t want = (size_t)(std::atof(argv[1]) * 1073741824.0);
    const size_t chunk = size_t(64) << 20;
    while (used < want) {
      void* p = nullptr;
      if (hipMalloc(&p, chunk) != hipSuccess) { (void)hipGetLastError(); break; }
      v.push_back(p);
      used += chunk;
    }
    for (void* p : v) (void)hipFree(p);
    std::printf("vram_perturb allocs=%zu GiB=%.2f\n", v.size(), used / 1073741824.0);
    return 0;
  }
  for (size_t chunk : {size_t(1) << 30, size_t(64) << 20}) {
    for (;;) {
      void* p = nullptr;
      if (hipMalloc(&p, chunk) != hipSuccess) { (void)hipGetLastError(); break; }
      v.push_back(p);
      used += chunk;
    }
  }
  for (void* p : v) (void)hipFree(p);
  if (v.empty()) { std::fprintf(stderr, "vram_reset: no allocation succeeded\n"); return 2; }
  std::printf("vram_reset allocs=%zu GiB=%.2f\n", v.size(), used / 1073741824.0);
  return 0;
}
