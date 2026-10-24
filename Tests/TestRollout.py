import torch

from supermimic.Heightmap import Heightmap, TerrainSpec
from supermimic.Rollout import predict_sequence, terrain_step
from supermimic.mdp import flatten_state
from supermimic.networks import Critic, Policy, WorldModel


def test_imagination_requeries_predicted_pose_and_reaches_actor_through_frozen_models():
    torch.manual_seed(19)
    S = (torch.tensor([[0., 0., .8], [.1, 0., .8]]), torch.eye(3).repeat(2, 1, 1),
         torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 29), torch.zeros(2, 29))
    heightmap = Heightmap(TerrainSpec(bounds=(-2., 2., -2., 2.)))
    wm = WorldModel(hidden=(24, 24))
    pi = Policy(6, 9, hist=2, hidden=(24, 16))
    critic = Critic(3, hidden=(16,))
    # A fitted dynamics head is nonzero; initialize one deterministically here.
    with torch.no_grad():
        wm.net[-1].weight.normal_(0, .01)
        wm.net[-1].bias[0] = .1
    for module in (wm, critic):
        module.requires_grad_(False)
    saved = [p.clone() for module in (wm, critic) for p in module.parameters()]
    history, goal, xi = torch.zeros(2, 2, 6), torch.zeros(2, 9), torch.zeros(2, 33)
    queried = []
    hook = heightmap.register_forward_pre_hook(lambda module, inputs: queried.append(inputs[0].detach().clone()))
    actions = []
    for _ in range(3):
        S, a = terrain_step(wm, pi, heightmap, S, history, xi, goal)
        actions.append(a)
    hook.remove()
    assert len(queried) == 3
    assert not torch.allclose(queried[0], queried[1])
    objective = S[4].square().mean() + critic(S[0], heightmap(S[0], S[1])).mean()
    optimizer = torch.optim.Adam(pi.parameters(), lr=.001)
    before = [p.clone() for p in pi.parameters()]
    optimizer.zero_grad()
    objective.backward()
    grads = [p.grad for p in pi.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pi.map_encoder.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, pi.parameters()))
    assert all(p.grad is None for module in (wm, critic) for p in module.parameters())
    assert all(torch.equal(a, b) for a, b in zip(saved, [p for module in (wm, critic) for p in module.parameters()]))
    # Real replay fitting uses recorded actions, while each state/map comes from
    # the autoregressive prediction; it is not a cache of old map embeddings.
    x = flatten_state((queried[0], torch.eye(3).repeat(2, 1, 1),
                       torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 29), torch.zeros(2, 29)))
    result = predict_sequence(wm, heightmap, x[:, None].expand(-1, 3, -1),
                              torch.stack(actions, 1).detach(), torch.zeros(2, 3, 33))
    assert result[0].shape == (2, 3, 3)
    assert torch.isfinite(result[0]).all()


def test_second_map_query_backpropagates_to_first_action_with_the_correct_slope():
    """Catch a pose detach even when the ordinary state/action path still works."""
    torch.manual_seed(23)
    S = (torch.tensor([[.03, .07, .8]]), torch.eye(3)[None],
         torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 29), torch.zeros(1, 29))
    heightmap = Heightmap(TerrainSpec(bounds=(-2., 2., -2., 2.)))
    # Analytic world surface z=.2*x; the centre pixel ignores orientation.
    heightmap.heights.copy_((.2 * torch.linspace(-2., 2., 81))[None, None, None, :].expand_as(heightmap.heights))
    wm = WorldModel(hidden=(24, 24))
    pi = Policy(6, 9, hist=2, hidden=(24, 16))
    with torch.no_grad():
        wm.net[-1].weight.normal_(0, .01)
    wm.requires_grad_(False)
    history, goal, xi = torch.zeros(1, 2, 6), torch.zeros(1, 9), torch.zeros(1, 33)
    maps = []
    hook = heightmap.register_forward_hook(lambda module, inputs, output: maps.append(output))
    try:
        S1, first_action = terrain_step(wm, pi, heightmap, S, history, xi, goal)
        terrain_step(wm, pi, heightmap, S1, history, xi, goal)
    finally:
        hook.remove()
    assert len(maps) == 2
    query_loss = maps[1][:, 0, 7, 11].sum()
    assert query_loss.requires_grad, "The second map lost its dependence on the first action"
    actual = torch.autograd.grad(query_loss, first_action, retain_graph=True)[0]
    # Independent expectation from the analytic terrain, not the query helper.
    expected = torch.autograd.grad((.2 * S1[0][:, 0] - S1[0][:, 2]).sum(), first_action)[0]
    assert torch.isfinite(actual).all()
    assert actual.abs().sum() > 1e-7
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_value_regression_updates_the_critics_own_map_encoder():
    torch.manual_seed(29)
    heightmap = Heightmap(TerrainSpec(bounds=(-2., 2., -2., 2.)))
    positions = torch.tensor([[0., 0., .8], [.1, .1, .9], [.2, 0., 1.0]])
    local = heightmap(positions, torch.eye(3).repeat(3, 1, 1))
    critic = Critic(3, hidden=(24, 16))
    optimizer = torch.optim.Adam(critic.parameters(), lr=.001)
    before = [p.detach().clone() for p in critic.map_encoder.parameters()]
    targets = torch.tensor([.3, -.2, .5])
    loss = (critic(positions, local) - targets).square().mean()
    optimizer.zero_grad()
    loss.backward()
    grads = [p.grad for p in critic.map_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, critic.map_encoder.parameters()))


def test_recorded_action_sequence_fitting_updates_world_model_map_encoder():
    torch.manual_seed(31)
    S = (torch.tensor([[0., 0., .8], [.1, 0., .9]]), torch.eye(3).repeat(2, 1, 1),
         torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 29), torch.zeros(2, 29))
    heightmap = Heightmap(TerrainSpec(bounds=(-2., 2., -2., 2.)))
    wm = WorldModel(hidden=(24, 24))
    with torch.no_grad():
        wm.net[-1].weight.normal_(0, .01)
    optimizer = torch.optim.Adam(wm.parameters(), lr=.001)
    before = [p.detach().clone() for p in wm.map_encoder.parameters()]
    states = flatten_state(S)[:, None].repeat(1, 3, 1)
    recorded_actions = torch.linspace(-.1, .1, 2 * 3 * 29).reshape(2, 3, 29)
    predicted = predict_sequence(wm, heightmap, states, recorded_actions, torch.zeros(2, 3, 33))
    targets = tuple(s[:, None].repeat(1, 3, *([1] * (s.ndim - 1))) for s in S)
    targets[0][:, :, 0] += torch.tensor([.03, .06, .09])
    loss = WorldModel.state_error(predicted, targets).square().mean()
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    grads = [p.grad for p in wm.map_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, wm.map_encoder.parameters()))
