#!/usr/bin/env bash
# frontier_prepare.sh -- GPA-Benchmark setup for Frontier (AMD MI250X, gfx90a, ROCm 7.0.2).
#
# Phases (default: all, in this order; each one is idempotent):
#   data    download + unpack the Rodinia data tarball into rodinia/data if it is absent.
#           Needs the internet: run it on a LOGIN node (compute nodes have none), or point
#           GPA_DATA_TARBALL at a local copy of data.tar.gz.
#   graph   build scripts/graphgen_seeded.cpp (fixed seed) and write rodinia/data/bfs/graph8M.txt
#           unless it already matches frontier_refs.md5. CPU only.
#   build   clean-build the 9 Frontier baselines (driver_apps.frontier.yaml) in place with hipcc.
#   refs    generate the references that are not tracked (bfs, pathfinder, b+tree with the
#           Frontier command file) from the pristine -hip baselines into frontier_refs/<app>/
#           (outside every app run dir), unless they already match frontier_refs.md5. Needs a
#           GPU (compute node).
#   verify  md5-check every tracked and generated reference and input (frontier_refs.md5).
#
# Usage: bash scripts/frontier_prepare.sh [data|graph|build|refs|verify ...]
# Env:   GPA_DATA_TARBALL=<path>  use a local data.tar.gz instead of downloading
#        ROCM_PATH / HIPCC / OFFLOAD_ARCH  toolchain (default /opt/rocm-7.0.2, gfx90a)
#        GPA_GRAPH_SEED=<n>       graphgen seed (default: the one compiled into graphgen_seeded.cpp;
#                                 any other seed will NOT match frontier_refs.md5)
set -euo pipefail
ulimit -c 0   # a crashing app must not leave a core file on Lustre

GPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GPA_ROOT"
MANIFEST="$GPA_ROOT/frontier_refs.md5"
DATA_URL="https://github.com/Jokeren/GPA-Benchmark/releases/download/datav0.1/data.tar.gz"

log()  { printf '[frontier_prepare] %s\n' "$*"; }
die()  { printf '[frontier_prepare] ERROR: %s\n' "$*" >&2; exit 1; }

_TMPDIRS=()
cleanup() {
    local rc=$? d
    for d in "${_TMPDIRS[@]:-}"; do
        if [[ -n "$d" ]]; then rm -rf "$d"; fi
    done
    return "$rc"
}
trap cleanup EXIT

# Frontier compute nodes are frontierNNNNN; login nodes (loginNN) have /dev/kfd too, so it is no signal.
is_compute_node() { [[ -n "${SLURM_JOB_ID:-}" || "$(hostname -s)" =~ ^frontier[0-9]+$ ]]; }

# Expected md5 for a manifest path ("" if absent).
expected_md5() { awk -v p="$1" '$2 == p { print $1 }' "$MANIFEST"; }
matches_manifest() {
    local p="$1" want
    want="$(expected_md5 "$p")"
    [[ -n "$want" && -f "$p" ]] || return 1
    [[ "$(md5sum "$p" | awk '{print $1}')" == "$want" ]]
}

setup_toolchain() {
    if [[ "${ROCM_PATH:-}" != */rocm-7.0.2 ]] && type module >/dev/null 2>&1; then
        module load rocm/7.0.2 >/dev/null 2>&1 || true
    fi
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.0.2}"
    export HIPCC="${HIPCC:-$ROCM_PATH/bin/hipcc}"
    export OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx90a}"
    [[ -x "$HIPCC" ]] || die "hipcc not found at $HIPCC (set ROCM_PATH or load rocm/7.0.2)"
    case "$ROCM_PATH" in
        */rocm-7.0.2) ;;
        *) log "WARNING: ROCM_PATH=$ROCM_PATH is not ROCm 7.0.2; the APPEB Frontier profilers refuse such binaries" ;;
    esac
    export PATH="$ROCM_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "toolchain: HIPCC=$HIPCC OFFLOAD_ARCH=$OFFLOAD_ARCH"
}

require_gpu() {
    local ri="${ROCM_PATH:-/opt/rocm-7.0.2}/bin/rocminfo"
    local agents
    agents="$("$ri" 2>/dev/null || true)"   # no pipe into grep -q: pipefail + SIGPIPE
    if ! is_compute_node || ! grep -q 'Name: *gfx' <<<"$agents"; then
        die "phase '$1' needs an AMD GPU: run it on a compute node"
    fi
}

# ---------------------------------------------------------------------------------------------
phase_data() {
    if [[ -d rodinia/data ]]; then
        log "data: rodinia/data exists"
        return 0
    fi
    local tmp; tmp="$(mktemp -d "$GPA_ROOT/.data_unpack.XXXXXX")"; _TMPDIRS+=("$tmp")
    if [[ -n "${GPA_DATA_TARBALL:-}" ]]; then
        [[ -f "$GPA_DATA_TARBALL" ]] || die "GPA_DATA_TARBALL=$GPA_DATA_TARBALL does not exist"
        log "data: unpacking $GPA_DATA_TARBALL"
        tar -xf "$GPA_DATA_TARBALL" -C "$tmp"
    else
        if is_compute_node && [[ -z "${https_proxy:-}${HTTPS_PROXY:-}" ]]; then
            die "rodinia/data is missing and this is a compute node (no internet). Run
    bash $GPA_ROOT/scripts/frontier_prepare.sh data
on a login node first, or set GPA_DATA_TARBALL=<local data.tar.gz>."
        fi
        log "data: downloading $DATA_URL"
        if command -v curl >/dev/null 2>&1; then
            curl -fL --retry 3 --connect-timeout 30 -o "$tmp/data.tar.gz" "$DATA_URL" \
                || die "download failed (no internet here? run the data phase on a login node)"
        else
            wget -q -O "$tmp/data.tar.gz" "$DATA_URL" \
                || die "download failed (no internet here? run the data phase on a login node)"
        fi
        tar -xf "$tmp/data.tar.gz" -C "$tmp"
        rm -f "$tmp/data.tar.gz"
    fi
    [[ -d "$tmp/data" ]] || die "tarball has no top-level data/ directory"
    mv "$tmp/data" rodinia/data
    log "data: rodinia/data ready"
}

phase_graph() {
    local out="rodinia/data/bfs/graph8M.txt"
    [[ -d rodinia/data/bfs ]] || die "rodinia/data/bfs missing: run the data phase first"
    local tmp; tmp="$(mktemp -d "$GPA_ROOT/.graphgen.XXXXXX")"; _TMPDIRS+=("$tmp")
    g++ -O2 -std=c++11 -o "$tmp/graphgen_seeded" scripts/graphgen_seeded.cpp
    if matches_manifest "$out"; then
        log "graph: $out matches frontier_refs.md5"
    else
        [[ -f "$out" ]] && log "graph: $out exists but does not match; regenerating"
        log "graph: generating 8388608-node graph (seeded)"
        (cd "$tmp" && ./graphgen_seeded 8388608 8M ${GPA_GRAPH_SEED:+"$GPA_GRAPH_SEED"} >/dev/null)
        mv "$tmp/graph8M.txt" "$out"
        if ! matches_manifest "$out"; then
            mv "$out" "$out.bad"
            die "graph: generated graph does not match frontier_refs.md5 (moved to $out.bad)"
        fi
        log "graph: $out generated and verified"
    fi
    # H7: a pool of seeded variant graphs for gpa_test's per-call random-input check (always).
    local pooldir="frontier_refs/bfs_variants"
    mkdir -p "$pooldir"
    for seed in 911 922 933 944; do
        local pout="$pooldir/graph8M_v${seed}.txt"
        matches_manifest "$pout" && { log "graph: $pout matches"; continue; }
        [[ -f "$pout" && ! -s "$MANIFEST" ]] && { log "graph: $pout exists (no manifest yet)"; continue; }
        log "graph: generating pooled variant graph (seed $seed)"
        (cd "$tmp" && ./graphgen_seeded 8388608 "8M_v${seed}" "$seed" >/dev/null)
        mv "$tmp/graph8M_v${seed}.txt" "$pout"
        if grep -q "graph8M_v${seed}.txt" "$MANIFEST" 2>/dev/null && ! matches_manifest "$pout"; then
            mv "$pout" "$pout.bad"
            die "graph: pooled graph $pout does not match frontier_refs.md5 (moved to $pout.bad)"
        fi
    done
    log "graph: bfs variant pool ready"
}

# name|dir|binary  (the 9 Frontier baselines; paths as in driver_apps.frontier.yaml)
BASELINES=(
    "bfs|rodinia/bfs-hip|bfs"
    "backprop|rodinia/backprop-hip|backprop"
    "b+tree|rodinia/b+tree-hip|b+tree.out"
    "heartwall|rodinia/heartwall-hip|heartwall"
    "hotspot|rodinia/hotspot-hip|hotspot"
    "pathfinder|rodinia/pathfinder-hip|pathfinder"
    "nw|rodinia/nw-hip|needle"
    "streamcluster|rodinia/streamcluster-hip|sc_gpu"
    "xsbench|XSBench-hip|XSBench"
)

phase_build() {
    setup_toolchain
    local entry name dir bin
    for entry in "${BASELINES[@]}"; do
        IFS='|' read -r name dir bin <<<"$entry"
        log "build: $name ($dir)"
        make -C "$dir" clean >/dev/null 2>&1 || true
        if ! make -C "$dir" -j8 OFFLOAD_ARCH="$OFFLOAD_ARCH" >"$dir/.frontier_build.log" 2>&1; then
            tail -30 "$dir/.frontier_build.log" >&2
            die "build of $name failed (log: $dir/.frontier_build.log)"
        fi
        [[ -x "$dir/$bin" ]] || die "build of $name produced no $dir/$bin"
    done
    log "build: 9 baselines built"
}

# gen_ref <manifest path> <app dir> <binary> <produced file or -stdout> <args...>
gen_ref() {
    local ref="$1" dir="$2" bin="$3" produced="$4"; shift 4
    if matches_manifest "$ref"; then
        log "refs: $ref matches frontier_refs.md5"
        return 0
    fi
    require_gpu refs
    [[ -x "$dir/$bin" ]] || die "refs: $dir/$bin not built (run the build phase)"
    log "refs: generating $ref"
    mkdir -p "$(dirname "$ref")"
    if [[ "$produced" == "-stdout" ]]; then
        (cd "$dir" && "./$bin" "$@" </dev/null >"$GPA_ROOT/$ref.tmp")
    else
        rm -f "$dir/$produced"
        (cd "$dir" && "./$bin" "$@" </dev/null >/dev/null)
        mv "$dir/$produced" "$ref.tmp"
    fi
    mv "$ref.tmp" "$ref"
    if ! matches_manifest "$ref"; then
        mv "$ref" "$ref.bad"
        die "refs: $ref does not match frontier_refs.md5 (moved to $ref.bad)"
    fi
}

phase_refs() {
    setup_toolchain
    # References live in frontier_refs/<app>/, never inside a directory an app runs in (R5).
    gen_ref frontier_refs/bfs/ref-result.txt rodinia/bfs-hip bfs result.txt \
        ../data/bfs/graph8M.txt
    gen_ref frontier_refs/pathfinder/ref-result.txt rodinia/pathfinder-hip pathfinder -stdout \
        300000 300 20
    gen_ref frontier_refs/b+tree/ref-output.txt rodinia/b+tree-hip b+tree.out output.txt \
        file ../data/b+tree/mil.txt command ./command_frontier.txt
    log "refs: generated references verified"
}

phase_verify() {
    [[ -f "$MANIFEST" ]] || die "missing $MANIFEST"
    if ! md5sum --quiet -c "$MANIFEST"; then
        die "verify: some references/inputs do not match frontier_refs.md5"
    fi
    log "verify: all $(grep -c . "$MANIFEST") entries of frontier_refs.md5 match"
}

phases=("$@")
[[ ${#phases[@]} -gt 0 ]] || phases=(data graph build refs verify)
for p in "${phases[@]}"; do
    case "$p" in
        data|graph|build|refs|verify) "phase_$p" ;;
        *) die "unknown phase '$p' (data|graph|build|refs|verify)" ;;
    esac
done
