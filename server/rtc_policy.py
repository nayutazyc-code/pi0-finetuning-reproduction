"""Original-pi0 JAX RTC inference adapter, pinned to the project's openpi revision.

Implements soft prefix inpainting from arXiv:2506.07339v2, Eq. 2--5.
The openpi reverse-time convention is t=1 noise, t=0 clean actions.
KV-cache and suffix evaluation adapt the pinned openpi pi0.py (Apache-2.0);
see repos/openpi/LICENSE. No parameters or upstream source are changed.
Private Policy attributes are
intentional: this adapter is supported only with the checked pinned revision.
"""
import types
import time

import einops
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils

SCHEMA = "pi0-rtc-v1"


def sample_actions_rtc(self, rng, observation, prefix, weights, beta, *, num_steps=10):
    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch = observation.state.shape[0]
    noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
    tokens, mask, ar_mask = self.embed_prefix(observation)
    _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask),
                                positions=jnp.cumsum(mask, axis=1) - 1)

    def velocity(x, t):
        suffix, smask, sar, cond = self.embed_suffix(observation, x, jnp.broadcast_to(t, batch))
        full_mask = jnp.concatenate((einops.repeat(mask, "b p -> b s p", s=suffix.shape[1]),
                                     make_attn_mask(smask, sar)), axis=-1)
        positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
        (_, output), _ = self.PaliGemma.llm([None, suffix], mask=full_mask, positions=positions,
                                           kv_cache=cache, adarms_cond=[None, cond])
        return self.action_out_proj(output[:, -self.action_horizon:])

    def step(index, x):
        t = 1.0 - index / num_steps
        def estimate(z):
            v = velocity(z, t)
            return z - t * v, v
        clean, pullback, v = jax.vjp(estimate, x, has_aux=True)
        residual = jax.lax.stop_gradient((prefix - clean) * weights)
        correction = pullback(residual)[0]
        tau = 1.0 - t
        # Eq. 2 rewritten for reverse time; the endpoint is bounded by beta.
        gain = jnp.minimum(beta, (t*t + tau*tau) / jnp.maximum(t*tau, 1e-6))
        return x - (v - gain * correction) / num_steps

    return jax.lax.fori_loop(0, num_steps, step, noise)


class RTCPolicy:
    def __init__(self, policy):
        if policy._is_pytorch_model or getattr(policy._model, 'pi05', False):
            raise ValueError("RTC adapter requires the pinned original-pi0 JAX model")
        self.base = policy
        method = types.MethodType(sample_actions_rtc, policy._model)
        self.sampler = nnx_utils.module_jit(method, static_argnames=("num_steps",))
        self.metadata = policy.metadata

    def infer(self, obs):
        request = obs.get("rtc")
        if request is None:
            return self.base.infer(obs)
        if request.get("schema") != SCHEMA:
            raise ValueError("Unsupported RTC schema")
        prefix = np.asarray(request["prefix"], np.float32)
        horizon = self.base._model.action_horizon
        valid, delay, blend = (request[k] for k in ("valid_steps", "delay_steps", "blend_steps"))
        if any(type(x) is not int for x in (valid, delay, blend)) or not 1 <= valid <= horizon:
            raise ValueError("Invalid RTC prefix length")
        if not 0 <= delay <= valid or not 0 <= blend <= horizon:
            raise ValueError("RTC delay/blend outside prefix horizon")
        beta = float(request["guidance_max"])
        if not np.isfinite(beta) or not 0 < beta <= 10:
            raise ValueError("RTC guidance_max must be in (0,10]")
        if prefix.shape != (horizon, 14) or not np.isfinite(prefix).all():
            raise ValueError("RTC prefix must be finite native ALOHA Hx14 targets")
        if np.any((prefix[:, [6, 13]] < 0) | (prefix[:, [6, 13]] > 1)):
            raise ValueError("RTC grippers must be native normalized [0,1] targets")
        end = min(valid, delay + blend)
        i = np.arange(horizon)
        c = np.clip((end - i) / max(1, end - delay + 1), 0, 1)
        temporal = np.where(i < delay, 1.0, c * np.expm1(c) / np.expm1(1.0))
        temporal[i >= end] = 0
        # Transform the prefix through exactly the serving policy's input transforms:
        # ALOHA conventions -> checkpoint normalization -> 32-dimensional padding.
        inputs = jax.tree.map(lambda x: x, {k: v for k, v in obs.items() if k != "rtc"})
        inputs["actions"] = prefix.copy()
        inputs = self.base._input_transform(inputs)
        transformed_prefix = inputs.pop("actions")
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        weights = np.zeros((1, horizon, self.base._model.action_dim), np.float32)
        weights[0, :, :14] = temporal[:, None]
        self.base._rng, rng = jax.random.split(self.base._rng)
        start = time.monotonic()
        actions = self.sampler(rng, model_lib.Observation.from_dict(inputs),
                               jnp.asarray(transformed_prefix)[None], jnp.asarray(weights),
                               jnp.asarray(beta, jnp.float32), **self.base._sample_kwargs)
        actions = np.asarray(actions[0])  # synchronize before reporting elapsed time
        result = self.base._output_transform({"state": np.asarray(inputs["state"][0]), "actions": actions})
        result["policy_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
        result["rtc"] = dict(schema=SCHEMA, guidance="clean_estimate_vjp", valid_steps=valid,
                             delay_steps=delay, constrained_steps=end, guidance_max=beta,
                             native_dimensions=14, padded_dimensions_guided=False)
        return result
