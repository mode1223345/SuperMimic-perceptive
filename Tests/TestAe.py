import importlib
import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec('supermimic.Ae') is not None, 'AE pretraining module is missing'
    return importlib.import_module('supermimic.Ae')


def test_ae_reconstructs_both_map_channels_and_learns_geometry():
    ae = api()
    torch.manual_seed(9)
    model = ae.Autoencoder(32)
    maps = ae.sample_maps(48, seed=4, split='train')
    assert model(maps).shape == maps.shape
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    first = ae.reconstruction_loss(model(maps), maps).item()
    for _ in range(100):
        loss = ae.reconstruction_loss(model(maps), maps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert ae.reconstruction_loss(model(maps), maps).item() < first * .15


def test_invalid_heights_do_not_contribute_to_reconstruction_loss():
    ae = api()
    target = torch.zeros(2, 2, 15, 23)
    pred = torch.zeros_like(target, requires_grad=True)
    baseline = ae.reconstruction_loss(pred, target)
    changed = target.clone()
    changed[:, 0] = 1000
    torch.testing.assert_close(ae.reconstruction_loss(pred, changed), baseline)
    baseline.backward()
    assert pred.grad[:, 0].abs().sum() == 0
    assert torch.isfinite(baseline)


def test_map_sampling_is_reproducible_with_separate_validation_geometry():
    ae = api()
    train = ae.sample_maps(64, seed=7, split='train')
    again = ae.sample_maps(64, seed=7, split='train')
    valid = ae.sample_maps(64, seed=7, split='validation')
    assert train.shape == (64, 2, 15, 23)
    assert torch.equal(train, again)
    assert not torch.equal(train, valid)
    assert torch.isfinite(train).all()
    assert ((train[:, 1] == 0) | (train[:, 1] == 1)).all()
    assert (train[:, 1] == 0).any() and (train[:, 1] == 1).any()
    assert train[:, 0].std() > .05
    with pytest.raises(ValueError):
        ae.sample_maps(0)


def test_encoder_artifact_roundtrip_is_frozen_and_detects_tampering(tmp_path):
    ae = api()
    model = ae.Autoencoder(32)
    path = tmp_path / 'Encoder.pt'
    info = ae.save_encoder(path, model.encoder, {'seed': 3})
    frozen, loaded_info = ae.load_encoder(path, map_dim=32)
    assert info == loaded_info
    maps = ae.sample_maps(4)
    torch.testing.assert_close(model.encoder(maps), frozen(maps))
    assert all(not p.requires_grad for p in frozen.parameters())
    with pytest.raises(ValueError, match='dimension'):
        ae.load_encoder(path, map_dim=16)
    payload = torch.load(path, weights_only=False)
    payload['encoder']['head.1.weight'][0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='hash'):
        ae.load_encoder(path, map_dim=32)
