#!/usr/bin/env bash
# Idempotent activation-time fix for a conda-forge numpy packaging quirk on
# this repo's platform (observed on numpy 1.26.4 build py310hb13e2d6_0, the
# MKL-linked build pixi's solver picks by default): that build ships a REAL,
# compiled numpy/_core/_multiarray_umath*.so alongside the genuine
# numpy/core/ implementation. Real PyPI numpy 1.26.x only ships a thin
# *pure-Python* numpy/_core/_multiarray_umath.py stub that re-exports from
# numpy/core/ (numpy._core is a NumPy-2.0-forward-compat alias, added as
# prep for the 2.0 migration -- it isn't supposed to have its own compiled
# implementation on the 1.26.x line). Because a compiled extension module
# shadows a same-named .py file, Python loads the stray .so instead of the
# thin stub.
#
# scipy>=1.15 (this repo's [pypi-dependencies] scipy, needed for
# bspline_subgoal_decomp.py's generate_knots) is built against numpy>=2.0's
# `numpy._core` namespace, so it picks up that stray compiled binary to
# build its special-function gufuncs (sph_legendre_p & co.). The resulting
# ufunc objects are a genuinely different `PyUFunc_Type` than the one numpy
# itself exposes as `numpy.ufunc` (via the real numpy/core/ path), so
# `isinstance(ufunc, numpy.ufunc)` is False even though both print as
# `<ufunc ...>`. That surfaces as, on any scipy.special/signal/interpolate
# import:
#   ValueError: All ufuncs must have type `numpy.ufunc`. Received (...)
#
# `pip install --force-reinstall numpy` alone does NOT fix this: pip only
# overwrites files that are part of the wheel it's installing (the thin
# stubs), it doesn't know to delete conda-installed files outside its own
# RECORD (the stray .so). Deleting numpy/_core/ first, then reinstalling,
# does -- that's what this script does.
#
# IMPORTANT: pixi sources [activation].scripts into its own activation shell
# (so env vars they set can persist) -- NOT run as a subprocess. That means
# this file must never `exit`/use `set -e`: doing so aborts pixi's own
# activation chain (including the PATH setup that comes after user scripts),
# which is exactly what happened the first version of this script shipped --
# it made `pixi run python` fail with "python: command not found". Everything
# below is written as plain conditionals with no early return, so a failure
# here can never take down the rest of activation.
#
# Runs on every `pixi run`/`pixi shell` activation. Cheap no-op after the
# first fix: it only touches anything if it finds the tell-tale stray file.

if [ -n "${CONDA_PREFIX:-}" ]; then
    _numpy_core_dir="$CONDA_PREFIX/lib/python3.10/site-packages/numpy/_core"
    if [ -d "$_numpy_core_dir" ]; then
        _stray_so="$(find "$_numpy_core_dir" -maxdepth 1 -name '_multiarray_umath*.so' -print -quit 2>/dev/null)"
        if [ -n "$_stray_so" ]; then
            echo "[pixi activation] Removing stray conda-forge numpy/_core compiled binary (scipy ufunc ABI fix)..." >&2
            _numpy_version="$("$CONDA_PREFIX/bin/python" -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"
            if [ -n "$_numpy_version" ]; then
                rm -rf "$_numpy_core_dir"
                "$CONDA_PREFIX/bin/python" -m pip install --force-reinstall --no-deps --no-cache-dir --quiet \
                    "numpy==$_numpy_version" || echo "[pixi activation] numpy _core fix failed, continuing anyway" >&2
            fi
        fi
        unset _stray_so _numpy_version
    fi
    unset _numpy_core_dir
fi
