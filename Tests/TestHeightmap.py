"""Behavioral tests for terrain coordinates, missing data and pose gradients."""

import importlib
import json
import math

import pytest
import torch


def module():
    try:
        return importlib.import_module("supermimic.Heightmap")
    except ModuleNotFoundError as exc:
        if exc.name == "supermimic.Heightmap":
            pytest.fail("The differentiable heightmap module is not implemented yet")
        raise


def yaw_rotation(theta):
    c, s = theta.cos(), theta.sin()
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one)).reshape(1, 3, 3)


def sloped_map():
    hm = module()
    terrain = hm.Heightmap(hm.TerrainSpec(bounds=(-1, 1, -1, 1)))
    y, x = torch.meshgrid(torch.linspace(-1, 1, 41), torch.linspace(-1, 1, 41), indexing="ij")
    terrain.heights.copy_((0.2 * x + 0.3 * y)[None, None])
    return terrain


def test_flat_map_has_base_relative_height_and_separate_validity():
    hm = module()
    terrain = hm.Heightmap(hm.TerrainSpec(bounds=(-1, 1, -1, 1)))
    result = terrain(torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.0, 1.1]]), torch.eye(3).repeat(2, 1, 1))
    assert result.shape == (2, 2, 15, 23)
    torch.testing.assert_close(result[:, 0, 7, 11], torch.tensor([-0.9, -1.1]))
    assert torch.all(result[:, 1] == 1)


def test_translations_and_yaw_preserve_world_surface_coordinates():
    terrain = sloped_map()
    p = torch.tensor([[0.1, 0.2, 0.9]])
    straight = terrain(p, torch.eye(3)[None])
    turned = terrain(p, yaw_rotation(torch.tensor(math.pi / 2)))
    torch.testing.assert_close(straight[0, 0, 7, 11], torch.tensor(-0.82))
    torch.testing.assert_close(straight[0, 0, 7, 13], torch.tensor(-0.80))
    torch.testing.assert_close(turned[0, 0, 7, 13], torch.tensor(-0.79))


def test_roll_and_pitch_do_not_tilt_the_sampling_grid():
    terrain = sloped_map()
    p = torch.tensor([[0.0, 0.0, 0.9]])
    pitch = torch.tensor([[math.cos(0.4), 0, math.sin(0.4)], [0, 1, 0], [-math.sin(0.4), 0, math.cos(0.4)]])
    roll = torch.tensor([[1, 0, 0], [0, math.cos(0.2), -math.sin(0.2)], [0, math.sin(0.2), math.cos(0.2)]])
    yaw = yaw_rotation(torch.tensor(0.3))
    torch.testing.assert_close(terrain(p, yaw), terrain(p, yaw @ pitch @ roll))


def test_out_of_bounds_and_missing_cells_are_invalid_not_flat_ground():
    hm = module()
    terrain = hm.Heightmap(hm.TerrainSpec(bounds=(-1, 1, -1, 1)))
    terrain.validity[0, 0, 20, 20] = 0
    result = terrain(torch.tensor([[0.0, 0.0, 0.9], [0.9, 0.0, 0.9]]), torch.eye(3).repeat(2, 1, 1))
    assert result[0, 1, 7, 11] == 0
    assert result[0, 0, 7, 11] == 0
    assert result[1, 1, 7, 22] == 0
    assert result[1, 0, 7, 22] == 0
    assert result[1, 1, 7, 11] == 1
    assert torch.isfinite(result).all()


def test_pose_and_yaw_gradients_follow_a_known_slope():
    terrain = sloped_map()
    p = torch.tensor([[0.03, 0.07, 0.9]], requires_grad=True)
    theta = torch.tensor(0.0, requires_grad=True)
    # At local (0.2, 0.1), dz/dyaw = -.2*.1 + .3*.2 = .04.
    height = terrain(p, yaw_rotation(theta))[0, 0, 9, 15]
    height.backward()
    torch.testing.assert_close(p.grad, torch.tensor([[0.2, 0.3, -1.0]]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(theta.grad, torch.tensor(0.04), rtol=1e-5, atol=1e-6)


def test_exact_boxes_and_checkpoint_roundtrip_preserve_terrain_identity():
    hm = module()
    spec = hm.TerrainSpec(profile="steps", bounds=(-1, 1, -1, 1), boxes=((0.0, 0.0, 0.5, 0.4, 0.08),))
    heights = spec.surface_height(torch.tensor([[0.0, 0.0], [0.24, 0.19], [0.26, 0.0]]))
    torch.testing.assert_close(heights, torch.tensor([0.08, 0.08, 0.0]))
    restored = hm.TerrainSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert spec == restored
    assert spec.fingerprint == restored.fingerprint
    changed = hm.TerrainSpec(profile="steps", bounds=(-1, 1, -1, 1), boxes=((0.0, 0.0, 0.5, 0.4, 0.09),))
    assert changed.fingerprint != spec.fingerprint


@pytest.mark.parametrize("kwargs", [
    {"cell_size": 0}, {"cell_size": float("nan")},
    {"bounds": (-1, 1.01, -1, 1)}, {"bounds": (1, -1, -1, 1)},
    {"profile": "unknown"}, {"profile": "steps", "boxes": ((0, 0, 0, 1, 0.1),)},
])
def test_invalid_terrain_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        module().TerrainSpec(**kwargs)


def test_encoder_retains_spatial_layout_and_gradients():
    hm = module()
    torch.manual_seed(11)
    encoder = hm.HeightEncoder(latent_dim=32)
    maps = torch.zeros(2, 2, 15, 23)
    maps[:, 1] = 1
    maps[0, 0, 5:8, 3:6] = 0.2
    maps[1, 0, 5:8, 17:20] = 0.2
    maps.requires_grad_()
    encoded = encoder(maps)
    assert encoded.shape == (2, 32)
    assert not torch.allclose(encoded[0], encoded[1], atol=1e-6)
    encoded.square().mean().backward()
    assert torch.isfinite(maps.grad).all()
    assert maps.grad[:, 0].abs().sum() > 0
