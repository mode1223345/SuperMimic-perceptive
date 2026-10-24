import importlib
import importlib.util

import pytest
import torch

from supermimic.Heightmap import Heightmap, TerrainSpec
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.Rollout import terrain_step


def api():
    assert importlib.util.find_spec('supermimic.Ae') is not None, 'Frozen shared encoder support is missing'
    return importlib.import_module('supermimic.Ae')


def models():
    ae = api()
    encoder = ae.freeze_encoder(ae.Autoencoder(32).encoder)
    pi = Policy(6, 9, hist=2, hidden=(32,), map_encoder=encoder)
    wm = WorldModel(hidden=(32,), map_encoder=encoder)
    critic = Critic(7, hidden=(32,), map_encoder=encoder)
    return ae, encoder, pi, wm, critic


def test_shared_encoder_is_one_object_and_survives_training_toggles():
    ae, encoder, pi, wm, critic = models()
    assert pi.map_encoder is wm.map_encoder is critic.map_encoder is encoder
    for model in (wm, critic, pi):
        ae.set_trainable(model, False)
        ae.set_trainable(model, True)
        model.train()
    ids = {id(p) for p in encoder.parameters()}
    for model in (pi, wm, critic):
        optimizer = torch.optim.Adam(ae.trainable_parameters(model))
        assert not ids.intersection(id(p) for group in optimizer.param_groups for p in group['params'])
    assert all(not p.requires_grad for p in encoder.parameters())


def test_frozen_encoder_retains_pose_gradient_and_does_not_change_after_updates():
    ae, encoder, pi, wm, critic = models()
    before = ae.encoder_hash(encoder)
    query = Heightmap(TerrainSpec(profile='steps', bounds=(-2, 4, -2, 2)))
    pos = torch.tensor([[1.2, 0., .8], [1.3, .1, .9]], requires_grad=True)
    image = query(pos, torch.eye(3).repeat(2, 1, 1))
    opt = torch.optim.Adam(ae.trainable_parameters(pi))
    loss = pi(torch.zeros(2, 2, 6), torch.zeros(2, 9), image).square().sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert pos.grad is not None and torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0
    assert all(p.grad is None for p in encoder.parameters())
    assert before == ae.encoder_hash(encoder)


def test_imagination_reuses_one_encoding_without_detaching():
    ae, encoder, pi, wm, _ = models()
    query = Heightmap(TerrainSpec())
    p = torch.tensor([[0., 0., .8]], requires_grad=True)
    S = (p, torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 29), torch.zeros(1, 29))
    calls = []
    handle = encoder.register_forward_hook(lambda *_: calls.append(1))
    out, action = terrain_step(wm, pi, query, S, torch.zeros(1, 2, 6), torch.zeros(1, 33), torch.zeros(1, 9))
    handle.remove()
    assert len(calls) == 1
    action.sum().backward()
    assert p.grad.abs().sum() > 0


def test_checkpoint_rejects_different_shared_embeddings():
    ae, encoder, pi, wm, critic = models()
    info = ae.encoder_info(encoder)
    ck = {'pi': pi.state_dict(), 'wm': wm.state_dict(), 'critic': critic.state_dict()}
    ae.validate_shared_checkpoint(ck, info)
    ck['wm'] = {k: v.clone() for k, v in ck['wm'].items()}
    ck['wm']['map_encoder.head.1.weight'][0, 0] += 1
    with pytest.raises(ValueError, match='encoder'):
        ae.validate_shared_checkpoint(ck, info)
