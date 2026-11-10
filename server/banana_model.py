"""Original pi0 with task-specific weighted Flow Matching loss; inference is inherited unchanged.
Core compute_loss derives from pinned openpi pi0.py (Apache-2.0); see repos/openpi/LICENSE.
"""
import dataclasses
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class BananaPi0Config(Pi0Config):
    dimension_weights: tuple[float, ...] = tuple([0.25]*7 + [1.0]*6 + [4.0] + [0.1]*18)
    image_augmentation: bool = False
    temporal_training: bool = False
    temporal_loss_weight: float = 0.0
    temporal_arm_scales: tuple[float, ...] = ()
    near_term_steps: int = 0
    near_term_weight: float = 1.0

    def create(self, rng):
        return BananaPi0(self, rngs=nnx.Rngs(rng))


class BananaPi0(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.dimension_weights = config.dimension_weights
        self.image_augmentation = config.image_augmentation
        self.temporal_training = config.temporal_training
        self.temporal_loss_weight = config.temporal_loss_weight
        self.temporal_arm_scales = config.temporal_arm_scales
        self.near_term_steps = config.near_term_steps
        self.near_term_weight = config.near_term_weight

    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_and_metrics(rng, observation, actions, train=train)
        return loss

    def compute_loss_and_metrics(self, rng, observation, actions, *, train=False):
        valid = getattr(observation, 'action_valid_mask', None)
        if self.temporal_training and valid is None:
            raise ValueError('Temporal training requires explicit episode-tail masks')
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train and self.image_augmentation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        weights = jnp.asarray(self.dimension_weights, dtype=jnp.float32)
        flow = jnp.sum(jnp.square(v_t - u_t) * weights, axis=-1) / jnp.sum(weights)
        if self.near_term_steps:
            from banana_temporal import near_term_flow_loss
            flow=near_term_flow_loss(flow,valid,self.near_term_steps,self.near_term_weight,xp=jnp)
        temporal = jnp.zeros(actions.shape[:-2], dtype=jnp.float32)
        if self.temporal_training:
            from banana_temporal import expert_difference_loss
            temporal = expert_difference_loss(x_t, v_t, actions, time, valid,
                                               self.temporal_arm_scales, xp=jnp)
        total = flow + self.temporal_loss_weight * temporal[..., None]
        return total, {'flow_loss': jnp.mean(flow), 'temporal_loss': jnp.mean(temporal),
                       'weighted_temporal_loss': self.temporal_loss_weight * jnp.mean(temporal)}
