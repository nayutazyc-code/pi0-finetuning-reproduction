"""Small deterministic NNX numerical diagnostic, separate from model training."""
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training.utils import TrainState
from banana_accumulation import install_accumulation


def run(output):
    class Tiny(nnx.Module):
        def __init__(self):
            self.action_out_proj = nnx.Linear(3, 2, rngs=nnx.Rngs(11))
        def compute_loss(self, rng, observation, actions, *, train=False):
            return jnp.square(self.action_out_proj(observation) - actions)

    model = Tiny()
    graph, params = nnx.split(model)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
    state = TrainState(step=0, params=params, model_def=graph, tx=tx, opt_state=tx.init(params),
                       ema_decay=None, ema_params=None)
    obs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    actions = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 7
    config = SimpleNamespace(trainable_filter=nnx.Param, model=SimpleNamespace(temporal_training=False))
    namespace = {'train_step': lambda *a: None}
    placeholder = FunctionType((lambda: None).__code__, namespace)
    accumulated = install_accumulation({'main': placeholder}, 2, 2)
    actual, info = jax.jit(lambda s: accumulated(config, jax.random.key(17), s, (obs, actions)))(state)
    jax.block_until_ready(actual)

    def loss_fn(model):
        return jnp.mean(model.compute_loss(jax.random.key(17), obs, actions, train=True))
    expected_loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    updates, _ = tx.update(grads, state.opt_state, params)
    expected = optax.apply_updates(params, updates)
    errors = [float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
              for a,b in zip(jax.tree.leaves(actual.params), jax.tree.leaves(expected))]
    record = dict(scope='Deterministic tiny NNX model; not a pi0 quality or GPU-memory measurement',
        micro_batch_size=2, accumulation_steps=2, effective_batch_size=4,
        optimizer_updates=int(actual.step), max_parameter_difference=max(errors),
        mean_loss_difference=float(abs(info['loss']-expected_loss)),
        gradient_norm_difference=float(abs(info['grad_norm']-optax.global_norm(grads))))
    record['passed'] = (record['optimizer_updates']==1 and record['max_parameter_difference']<1e-6
                         and record['mean_loss_difference']<1e-6 and record['gradient_norm_difference']<1e-5)
    Path(output).write_text(json.dumps(record, indent=2)+'\n')
    if not record['passed']:
        raise AssertionError(record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
