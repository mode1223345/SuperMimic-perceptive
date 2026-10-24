"""Behavioral tests for terrain-conditioned networks without an Isaac Lab installation."""

import torch

from supermimic.networks import Critic, Policy, WorldModel


def state(batch=4):
    return (torch.tensor([[0., 0., .8]]).repeat(batch, 1),
            torch.eye(3).repeat(batch, 1, 1), torch.zeros(batch, 3),
            torch.zeros(batch, 3), torch.zeros(batch, 29), torch.zeros(batch, 29))


def maps(batch=4):
    image = torch.zeros(batch, 2, 15, 23)
    image[:, 1] = 1
    image[batch // 2:, 0, :, 12:] = .15
    return image


def test_policy_and_value_distinguish_terrain_with_identical_body_inputs():
    torch.manual_seed(12)
    pi = Policy(6, 9, hist=2, hidden=(32, 16))
    critic = Critic(7, hidden=(32, 16))
    image = maps()
    action = pi(torch.zeros(4, 2, 6), torch.zeros(4, 9), image)
    value = critic(torch.zeros(4, 7), image)
    assert action.shape == (4, 29)
    assert (action[0] - action[2]).abs().max() > 1e-7
    assert (value[0] - value[2]).abs() > 1e-7
    assert not {id(p) for p in pi.map_encoder.parameters()} & {id(p) for p in critic.map_encoder.parameters()}


def test_world_model_learns_different_outcomes_for_the_same_body_and_action():
    torch.manual_seed(6)
    wm = WorldModel(hidden=(32, 32))
    S, image = state(), maps()
    target = list(state())
    target[4] = torch.full((4, 29), -.02)
    target[4][2:] = .02
    optim = torch.optim.Adam(wm.parameters(), lr=.001)
    before = [p.detach().clone() for p in wm.map_encoder.parameters()]
    losses = []
    for _ in range(80):
        predicted = wm.step(S, torch.zeros(4, 29), torch.zeros(4, 33), image)
        loss = (predicted[4] - target[4]).square().mean()
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert losses[-1] < losses[0] * .3
    assert any(not torch.equal(a, b) for a, b in zip(before, wm.map_encoder.parameters()))


def test_map_free_ablation_preserves_upstream_call_signature():
    wm = WorldModel(hidden=(16,), map_dim=0)
    pi = Policy(6, 9, hist=2, hidden=(16,), map_dim=0)
    critic = Critic(7, hidden=(16,), map_dim=0)
    assert wm.step(state(), torch.zeros(4, 29), torch.zeros(4, 33))[0].shape == (4, 3)
    assert pi(torch.zeros(4, 2, 6), torch.zeros(4, 9)).shape == (4, 29)
    assert critic(torch.zeros(4, 7)).shape == (4,)
