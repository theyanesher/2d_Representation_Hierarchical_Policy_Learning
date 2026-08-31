#!/usr/bin/env bash
# Post-reboot GPU health check.
#
# Run this after rebooting to confirm the 2026-08-28 driver-mismatch incident
# is fully resolved. Every line should read [ OK ]. Exits non-zero if not.
#
#   ./scripts/check_gpu_health.sh
REPO=/home/theyanesh/2d_Representation_Hierarchical_Policy_Learning
EVAL_PY="$REPO/.pixi/envs/eval/bin/python"
WT_DIR=/home/theyanesh/worktrees/theya_low_level_dit2d/Low_Level_and_Inference
FAIL=0
ok(){ printf '  [ OK ] %s\n' "$1"; }
bad(){ printf '  [FAIL] %s\n' "$1"; FAIL=1; }

echo "== 1. pending reboot =="
if [ -f /var/run/reboot-required ]; then
  bad "reboot still required: $(tr '\n' ' ' < /var/run/reboot-required.pkgs 2>/dev/null)"
else ok "no reboot pending"; fi

echo "== 2. kernel module vs userspace =="
KV=$(awk '{print $8}' /proc/driver/nvidia/version 2>/dev/null)
UV=$(basename "$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1)" | sed 's/libcuda.so.//')
echo "     kernel=$KV  userspace=$UV"
[ -n "$KV" ] && [ "$KV" = "$UV" ] && ok "versions match" || bad "MISMATCH (this was the bug)"

echo "== 3. shim removed =="
[ -d "$HOME/.local/opt/nvidia-580.173.02" ] \
  && bad "shim still present -- delete it, it will re-break things" \
  || ok "shim gone"
case ":${LD_LIBRARY_PATH:-}:" in *nvidia-580.173.02*)
  bad "shim still on LD_LIBRARY_PATH (stale shell? re-login)";; *) ok "LD_LIBRARY_PATH clean";; esac

echo "== 4. nvidia-smi (native) =="
if SMI=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1); then
  printf '       %s\n' "$SMI"; ok "nvidia-smi works"
else printf '       %s\n' "$SMI"; bad "nvidia-smi failed"; fi

echo "== 5. CUDA init in a fresh process =="
R=$("$EVAL_PY" -c "
import ctypes
l=ctypes.CDLL('libcuda.so.1'); r=l.cuInit(0)
n=ctypes.c_int(); l.cuDeviceGetCount(ctypes.byref(n))
print(r,n.value)" 2>/dev/null)
[ "$R" = "0 2" ] && ok "cuInit=0, 2 devices" || bad "cuInit/devices = '$R' (expect '0 2'; 804 = mismatch)"

echo "== 6. torch in eval env (render pipeline) =="
"$EVAL_PY" -c "
import torch
assert torch.cuda.is_available(), 'cuda not available'
x=torch.randn(512,512,device='cuda'); assert torch.isfinite((x@x).sum())
print('       devices:',torch.cuda.device_count(),torch.cuda.get_device_name(0))" 2>/dev/null \
  && ok "torch matmul on GPU" || bad "torch GPU failed in eval env"

echo "== 7. EGL renders on the NVIDIA GPU (not llvmpipe) =="
GLOUT=$(MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$EVAL_PY" -c "
import mujoco
m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"sphere\" size=\".3\"/></worldbody></mujoco>')
d=mujoco.MjData(m); mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,64,64); r.update_scene(d); img=r.render()
from OpenGL import GL
print('GLCHECK', GL.glGetString(GL.GL_VENDOR).decode(), '|', GL.glGetString(GL.GL_RENDERER).decode())
" 2>/dev/null | grep GLCHECK)
echo "       ${GLOUT:-<no GL string returned>}"
case "$GLOUT" in
  *NVIDIA*) ok "EGL bound to the NVIDIA GPU" ;;
  *llvmpipe*|*Mesa*|*softpipe*|*swrast*)
    bad "EGL fell back to SOFTWARE rendering -- renders would be ~100x slower and workers crash-loop" ;;
  *) bad "could not determine GL renderer" ;;
esac

echo "== 8. torch in worktree env (eval_approach2_rerun.py) =="
if [ -d "$WT_DIR" ]; then
  (cd "$WT_DIR" && timeout 300 pixi run python -c "
import torch; assert torch.cuda.is_available()
print('       devices:',torch.cuda.device_count())" 2>/dev/null) \
    && ok "torch GPU in worktree env" || bad "torch GPU failed in worktree env"
else ok "worktree not present, skipped"; fi

echo "== 9. DKMS built for running kernel =="
dkms status 2>/dev/null | grep -q "$(uname -r)" \
  && ok "dkms module present for $(uname -r)" || bad "no dkms module for $(uname -r)"

echo "== 10. stale driver branches =="
S=$(dpkg -l 2>/dev/null | grep -cE "nvidia.*(575|595)")
[ "$S" -eq 0 ] && ok "no 575/595 leftovers" \
  || echo "  [WARN] $S stale 575/595 packages -- latent future mismatch, purge when convenient"

echo
[ $FAIL -eq 0 ] && echo "ALL GPU CHECKS PASSED" || echo "SOME CHECKS FAILED (see [FAIL] above)"
exit $FAIL
