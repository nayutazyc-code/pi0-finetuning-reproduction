"""Single-GPU microbatch accumulation; one optimizer update per outer step.

The loader supplies an effective batch. Activations are evaluated sequentially
with lax.fori_loop, and gradients are averaged before clipping/AdamW/EMA.
No pinned upstream files are changed.
"""
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.shared import nnx_utils


def install_accumulation(training, micro_batch_size, accumulation_steps):
    namespace = training["main"].__globals__
    original = namespace["train_step"]
    if accumulation_steps == 1:
        return original
    if jax.device_count() != 1 or jax.process_count() != 1:
        raise ValueError("This accumulation adapter supports one GPU/process")

    def accumulated_step(config, rng, state, batch):
        observation, actions = batch
        if actions.shape[0] != micro_batch_size * accumulation_steps:
            raise ValueError("Effective batch does not match microbatch times accumulation")
        train_rng = jax.random.fold_in(rng, state.step)
        params = state.params.filter(config.trainable_filter)
        zeros = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
        temporal = getattr(config.model, 'temporal_training', False)

        def accumulate(index, carry):
            total_grads, total_loss, total_aux = carry
            start = index * micro_batch_size
            micro_observation = jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, start, micro_batch_size, axis=0), observation)
            micro_actions = jax.lax.dynamic_slice_in_dim(actions, start, micro_batch_size, axis=0)
            # Each loop body owns its NNX graph; no mutable model crosses the
            # lax transformation boundary. All microbatches see identical weights.
            model = nnx.merge(state.model_def, state.params)
            model.train()

            def loss_fn(model, key, obs, target):
                if temporal:
                    values, aux = model.compute_loss_and_metrics(key, obs, target, train=True)
                    return jnp.mean(values), jnp.stack([aux['flow_loss'], aux['temporal_loss'], aux['weighted_temporal_loss']])
                return jnp.mean(model.compute_loss(key, obs, target, train=True)), jnp.zeros(3)

            diff = nnx.DiffState(0, config.trainable_filter)
            (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
                model, jax.random.fold_in(train_rng, index), micro_observation, micro_actions)
            return jax.tree.map(lambda a, g: a + g.astype(jnp.float32), total_grads, grads), total_loss + loss, total_aux + aux

        grads, loss, aux = jax.lax.fori_loop(0, accumulation_steps, accumulate, (zeros, jnp.float32(0), jnp.zeros(3)))
        grads = jax.tree.map(lambda g: g / accumulation_steps, grads)
        loss = loss / accumulation_steps
        updates, opt_state = state.tx.update(grads, state.opt_state, params)
        model = nnx.merge(state.model_def, state.params)
        nnx.update(model, optax.apply_updates(params, updates))
        new_params = nnx.state(model)
        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=opt_state)
        if state.ema_decay is not None:
            new_state = dataclasses.replace(new_state, ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1-state.ema_decay) * new,
                state.ema_params, new_params))
        kernels = nnx.state(model, nnx.All(nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1))
        metrics = {"loss": loss, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(kernels)}
        if temporal:
            metrics.update(flow_loss=aux[0]/accumulation_steps, temporal_loss=aux[1]/accumulation_steps,
                           weighted_temporal_loss=aux[2]/accumulation_steps)
        return new_state, metrics

    namespace["train_step"] = accumulated_step
    return accumulated_step
