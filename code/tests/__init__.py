"""Test-suite-wide setup.

Configures a persistent JAX compilation cache so repeated ``python -m
unittest`` invocations (locally or in CI) skip recompiling XLA graphs seen in
a previous run. The trainer test files do real (if tiny -- n_steps=2..4)
training, and XLA JIT compilation dominates their wall time regardless of
step count: measured at ~50-58s per test, almost entirely compile, not the
training steps themselves. This is the highest-leverage, zero-coverage-impact
suite-speed change available -- it changes nothing about what is asserted.
See AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#68.

Respects JAX_COMPILATION_CACHE_DIR if the environment already sets one (e.g.
CI pointing it somewhere ephemeral); otherwise defaults to a directory under
the user's cache dir so it persists across invocations on the same machine.

Runs once, before any test submodule is imported (this __init__.py executes
first for every `python -m unittest tests.test_X` invocation, regardless of
which module is targeted). This is purely a speed optimization with zero
effect on what any test asserts, so every step is best-effort and non-fatal:
several test files (post-training-analysis, data-loader unit tests, ...) need
no JAX training stack at all and must stay importable/runnable whether jax is
missing, an incompatible version rejects one of these config keys, the cache
directory can't be created/written, or `import jax` itself raises (a
jax/jaxlib version mismatch typically surfaces as RuntimeError or OSError at
import time, not ImportError, so that alone isn't enough to catch it).
"""

import os

try:
    import jax
except Exception:
    # Anything from a plain missing package to a jax/jaxlib version mismatch
    # (RuntimeError, OSError, ...) -- none of it should be able to break
    # importing tests for files that need no JAX stack at all.
    jax = None

if jax is not None:
    try:
        _cache_dir = os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.expanduser("~/.cache/aind-dynamic-foraging-bfm-wrapper/jax"),
        )
        os.makedirs(_cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        # Cache every compile, even sub-second ones. All our real compiles
        # are multi-second, so this mainly guards against a future fast test
        # being silently excluded by JAX's default minimum-compile-time
        # threshold.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    except Exception:
        # Config-key or directory failures (older/newer JAX, unwritable
        # cache dir, ...) must never break importing the tests package --
        # tests just run without the cache, at the pre-#68 speed.
        pass
