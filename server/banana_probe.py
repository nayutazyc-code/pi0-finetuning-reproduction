"""Real-data, real-gradient GPU memory/throughput probe without checkpoint saves."""
import functools
import json
from pathlib import Path
import subprocess
import threading
import time

import jax
import numpy as np
from openpi.training import data_loader
from openpi.training import sharding


def run_probe(training, config, step_fn, updates, output, description):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status="initializing", description=description, updates_requested=updates,
                  scope="Disposable optimization steps for GPU measurement; no saved trained checkpoint", timings=[])
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                text = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                    '--format=csv,noheader,nounits'], text=True, timeout=5).strip().splitlines()[0]
                used, total = map(int, text.split(','))
                samples.append((time.time(), used, total))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(.2)

    def save():
        record['gpu_peak_used_mib'] = max((s[1] for s in samples), default=None)
        record['gpu_total_mib'] = samples[-1][2] if samples else None
        record['gpu_memory_samples'] = len(samples)
        output.write_text(json.dumps(record, indent=2) + '\n')

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    save()
    try:
        if jax.default_backend() != 'gpu' or jax.device_count() != 1:
            raise ValueError('Probe requires exactly one CUDA GPU')
        mesh = sharding.make_mesh(config.fsdp_devices)
        batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        loader = data_loader.create_data_loader(config, sharding=batch_sharding, shuffle=True)
        iterator = iter(loader)
        batch = next(iterator)
        record['actual_action_shape'] = list(batch[1].shape)
        key, init = jax.random.split(jax.random.key(config.seed))
        state, state_sharding = training['init_train_state'](config, init, mesh, resume=False)
        jax.block_until_ready(state)
        compiled = jax.jit(functools.partial(step_fn, config),
            in_shardings=(replicated, state_sharding, batch_sharding),
            out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        record['status'] = 'running'
        save()
        for index in range(updates):
            start = time.perf_counter()
            with sharding.set_mesh(mesh):
                state, info = compiled(key, state, batch)
            jax.block_until_ready(state)
            elapsed = time.perf_counter() - start
            metrics = {k: float(v) for k, v in jax.device_get(info).items()}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Non-finite probe loss/gradient')
            record['timings'].append(dict(optimizer_update=index+1, seconds=elapsed, metrics=metrics))
            record['completed_optimizer_updates'] = int(state.step)
            save()
            batch = next(iterator)
        record['steady_update_seconds'] = float(np.median([t['seconds'] for t in record['timings'][1:]]))
        record['estimated_training_hours'] = record['steady_update_seconds'] * config.num_train_steps / 3600
        record['jax_memory_stats'] = {k: int(v) for k,v in (jax.devices()[0].memory_stats() or {}).items()
                                      if isinstance(v, (int, np.integer))}
        record['status'] = 'passed'
    except BaseException as exc:
        record['status'] = 'failed'
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        save()
