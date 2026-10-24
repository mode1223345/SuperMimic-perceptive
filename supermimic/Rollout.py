"""Shared real-action fitting and imagined-action transitions (no simulator imports)."""

import torch

from .mdp import normalize_xi, unflatten_state


def predict_sequence(wm, heightmap, states, actions, contexts):
    """Start at a real state; unroll recorded actions using newly predicted maps."""
    S = unflatten_state(states[:, 0])
    predictions = []
    for j in range(actions.shape[1]):
        local = heightmap(S[0], S[1])
        S = wm.step(S, actions[:, j], normalize_xi(contexts[:, j]), local)
        predictions.append(S)
    return tuple(torch.stack([s[i] for s in predictions], 1) for i in range(6))


def terrain_step(wm, pi, heightmap, S, history, xi_normalized, goal):
    """Do not detach S/query: BPTT must reach earlier actions through map changes."""
    local = heightmap(S[0], S[1])
    if getattr(pi, 'map_encoder', None) is not None and pi.map_encoder is getattr(wm, 'map_encoder', None):
        latent = pi.map_encoder(local)
        action = pi(history, goal, map_latent=latent)
        return wm.step(S, action, xi_normalized, map_latent=latent), action
    action = pi(history, goal, local)
    return wm.step(S, action, xi_normalized, local), action
