#!/usr/bin/env bash
# Collect NCU profiles of LULESH ApplyMaterialPropertiesAndUpdateVolume_kernel
# over the Cartesian product of block sizes and max register counts.
# Run from repo root. Requires ncu (NVIDIA Nsight Compute) and CUDA.

set -e

SCRIPT_DIR="$(realpath $(dirname ${BASH_SOURCE[0]}))"
REPO_ROOT="$(realpath ${SCRIPT_DIR})"
LULESH_SRC="${REPO_ROOT}/LULESH/cuda/src"
PROFILE_DIR="${SCRIPT_DIR}/profiles/"
PROFILE_SUFFIX="A100"
SM_VERSION=80

# From driver_apps.yaml
CLEAN_CMD="make -f ../build/Makefile clean"
BUILD_CMD="make -f ../build/Makefile -j 8 SM_VERSION=${SM_VERSION}"
NCU_ARGS="-k ApplyMaterialPropertiesAndUpdateVolume_kernel --launch-skip 49 --launch-count 1 --metrics regex:sm__inst_executed_pipe_[^.]*.avg.pct_of_peak_sustained_active$,regex:sm__sass_thread_inst_executed_op.*sum$,regex:l1tex__t_set_.*_pipe_lsu_mem_global_op_ld.sum$,regex:l1tex__t_set_accesses.sum$,regex:l1tex__t_requests.sum$,regex:l1tex__m_xbar2l1tex_read_sectors.sum$,sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime,regex:sm__sass_inst_executed.*sum$,regex:sm__inst_issued.avg.per_cycle_active$,regex:.*throughput.avg.pct_of_peak_sustained_active$,regex:.*throughput.avg.pct_of_peak_sustained_elapsed$ --set full --import-source yes --target-processes all"
BLOCK_SIZES=(64 128 256 512)
REG_COUNTS=(32 48 64 96)

mkdir -p "$PROFILE_DIR"

echo "LULESH NCU kernel profiling: block sizes=${BLOCK_SIZES[*]}, reg counts=${REG_COUNTS[*]}"
echo "Profiles will be written to: $PROFILE_DIR"
echo ""

for reg_count in "${REG_COUNTS[@]}"; do
  echo "=== Building with REG_COUNT=$reg_count ==="
  cd "$LULESH_SRC"
  $CLEAN_CMD
  $BUILD_CMD REG_COUNT="$reg_count"

  for block_size in "${BLOCK_SIZES[@]}"; do
    out_name="lulesh_bs${block_size}_reg${reg_count}_${PROFILE_SUFFIX}"
    out_path="$PROFILE_DIR/$out_name"
    echo "  Profiling block_size=$block_size -> $out_name.ncu-rep"
    ncu $NCU_ARGS -o "$out_path" ./lulesh -s 45 -b "$block_size"
  done
  echo ""
done

echo "Done. Profiles in $PROFILE_DIR:"
ls -la "$PROFILE_DIR"
