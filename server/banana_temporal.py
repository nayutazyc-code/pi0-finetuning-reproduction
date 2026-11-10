"""Expert-difference supervision on clean estimates; no JAX import needed for audits."""
ARM_JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def near_term_flow_loss(per_step_flow, valid, prefix_steps, prefix_weight, *, xp):
    """Keep FM targets, emphasize upcoming valid actions with mean-one weights.

    No-action padding is explicitly excluded; real constant poses stay valid.
    Normalizing each example's weights avoids silently scaling its learning rate.
    This does not impose a grasp/phase rule or change the prediction horizon.
    """
    horizon=per_step_flow.shape[-1]
    weights=xp.where(xp.arange(horizon)<prefix_steps,prefix_weight,1.0)
    weights=xp.where(valid,weights,0.0)
    normalization=xp.maximum(xp.mean(weights,axis=-1,keepdims=True),1e-12)
    return per_step_flow*weights/normalization


def expert_difference_loss(noisy_actions, predicted_flow, actions, noise_time, valid, arm_scales, *, xp):
    """Per-example mean squared adjacent-displacement error in native radians.

    The pinned path is x_t=t*noise+(1-t)*actions, so clean=x_t-t*v.
    Means cancel in differences; normalization std+epsilon restores radians.
    ALOHA arm sign flips do not affect the squared error. Excludes grippers,
    padded model axes and invalid episode-tail pairs. Static expert poses remain
    valid targets; padding is never inferred from equal-valued actions.
    """
    clean = noisy_actions - noise_time[..., None, None] * predicted_flow
    error = (clean[..., 1:, :] - clean[..., :-1, :]) - (actions[..., 1:, :] - actions[..., :-1, :])
    error = xp.take(error, xp.asarray(ARM_JOINTS), axis=-1) * xp.asarray(arm_scales)
    pairs = valid[..., 1:] & valid[..., :-1]
    squared = xp.mean(xp.square(error), axis=-1)
    numerator = xp.sum(xp.where(pairs, squared, 0), axis=-1)
    denominator = xp.maximum(xp.sum(pairs, axis=-1), 1)
    return numerator / denominator
