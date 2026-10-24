"""Elevation AE, deterministic geometry data, and permanently frozen ownership.

Freezing parameters deliberately does NOT disable differentiation with respect
to the map input: imagination still needs pose -> map -> latent gradients.
"""
from pathlib import Path
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .Heightmap import HeightEncoder, Heightmap, TerrainSpec


class Autoencoder(nn.Module):
    def __init__(self, map_dim=32):
        super().__init__()
        self.encoder = HeightEncoder(map_dim)
        self.decoder = nn.Sequential(nn.Linear(map_dim, 128), nn.ELU(),
                                     nn.Linear(128, 256), nn.ELU(), nn.Linear(256, 690))

    def forward(self, maps):
        # Channel zero is metres; channel one is a validity logit.
        return self.decoder(self.encoder(maps)).reshape(-1, 2, 15, 23)


def reconstruction_loss(prediction, target):
    valid = target[:, 1]
    height = ((prediction[:, 0] - target[:, 0]).square() * valid).sum() / valid.sum().clamp_min(1)
    mask = F.binary_cross_entropy_with_logits(prediction[:, 1], valid)
    return height + .01 * mask


@torch.no_grad()
def reconstruction_metrics(prediction, target):
    height, valid = target[:, 0], target[:, 1]
    error = prediction[:, 0] - height
    count = valid.sum().clamp_min(1)
    edge = torch.zeros_like(valid, dtype=torch.bool)
    dx = (height[:, :, 1:] - height[:, :, :-1]).abs() > .01
    dx &= (valid[:, :, 1:] * valid[:, :, :-1]).bool()
    dy = (height[:, 1:, :] - height[:, :-1, :]).abs() > .01
    dy &= (valid[:, 1:, :] * valid[:, :-1, :]).bool()
    edge[:, :, 1:] |= dx
    edge[:, :, :-1] |= dx
    edge[:, 1:, :] |= dy
    edge[:, :-1, :] |= dy
    return {'height_rmse_m': float(((error.square() * valid).sum() / count).sqrt()),
            'height_mae_m': float((error.abs() * valid).sum() / count),
            'edge_rmse_m': float(((error.square() * edge).sum() / edge.sum().clamp_min(1)).sqrt()),
            'edge_pixels': int(edge.sum()),
            'validity_accuracy': float(((prediction[:, 1] >= 0) == valid.bool()).float().mean())}


def sample_maps(count, seed=0, split='train', device='cpu', profile=None):
    """Use the production raster/query. Validation holds out procedural layouts.

    Canonical Flat/Steps layouts appear in both splits with disjoint random pose
    samples. Other layouts use independently seeded geometry. No reward data.
    """
    if count <= 0 or split not in ('train', 'validation') or profile not in (None, 'flat', 'steps'):
        raise ValueError('Invalid map count, split or profile')
    rng = np.random.default_rng(seed + (100_003 if split == 'validation' else 0))
    groups = 1 if profile else min(count, 34 if split == 'train' else 10)
    outputs = []
    for group, indices in enumerate(np.array_split(np.arange(count), groups)):
        n = len(indices)
        kind = profile or ('flat' if group == 0 else 'steps')
        boxes = None
        if profile is None and group >= 2:
            boxes = tuple((float(rng.uniform(-1, 3)), float(rng.uniform(-1, 1)),
                           float(rng.uniform(.15, 1.2)), float(rng.uniform(.15, 1.2)),
                           float(rng.uniform(.02, .24))) for _ in range(int(rng.integers(1, 7))))
        spec = TerrainSpec(kind, bounds=(-2., 4., -2., 2.), boxes=boxes)
        query = Heightmap(spec, device)
        xy = rng.uniform([-1.4, -1.4], [3.4, 1.4], (n, 2))
        if spec.boxes:
            near = rng.random(n) < .7
            anchors = np.asarray(spec.boxes)[rng.integers(len(spec.boxes), size=n), :2]
            xy[near] = (anchors + rng.uniform(-.65, .65, (n, 2)))[near]
        # Include map-boundary views, so the validity channel is not constant.
        boundary = rng.random(n) < .15
        xy[boundary, 0] = rng.choice([-2., 4.], boundary.sum()) + rng.uniform(-.3, .3, boundary.sum())
        z = rng.uniform(.25, 1.25, n)
        extreme = rng.random(n) < .1
        z[extreme] = rng.uniform(.02, 1.6, extreme.sum())
        p = torch.tensor(np.column_stack((xy, z)), dtype=torch.float32, device=device)
        yaw = torch.tensor(rng.uniform(-math.pi, math.pi, n), dtype=torch.float32, device=device)
        R = torch.eye(3, device=device).repeat(n, 1, 1)
        R[:, 0, 0] = R[:, 1, 1] = yaw.cos()
        R[:, 1, 0], R[:, 0, 1] = yaw.sin(), -yaw.sin()
        with torch.no_grad():
            outputs.append(query(p, R))
    return torch.cat(outputs)


def encoder_hash(encoder):
    state = encoder.state_dict() if isinstance(encoder, nn.Module) else encoder
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{key}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def encoder_info(encoder):
    return {'kind': 'frozen_ae', 'version': 1, 'architecture': 'HeightEncoder-v1',
            'map_dim': encoder.head[3].out_features,
            'local_shape': [2, 15, 23], 'height': 'terrain_z-base_z',
            'alignment': 'gravity+yaw', 'sha256': encoder_hash(encoder)}


def freeze_encoder(encoder):
    encoder._frozen_map = True
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def set_trainable(model, flag):
    frozen = {id(p) for module in model.modules() if getattr(module, '_frozen_map', False)
              for p in module.parameters()}
    for p in model.parameters():
        p.requires_grad_(flag and id(p) not in frozen)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def save_encoder(path, encoder, pretraining):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = encoder_info(encoder)
    payload = {'encoder': {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
               'metadata': info, 'pretraining': pretraining}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return info


def load_encoder(path, map_dim=32, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    info = payload['metadata']
    if info['map_dim'] != map_dim:
        raise ValueError('AE latent dimension does not match --map_dim')
    if encoder_hash(payload['encoder']) != info['sha256']:
        raise ValueError('AE encoder hash mismatch')
    encoder = HeightEncoder(map_dim)
    encoder.load_state_dict(payload['encoder'])
    if encoder_info(encoder) != info:
        raise ValueError('AE encoder metadata/semantics mismatch')
    return freeze_encoder(encoder.to(device)), info


def validate_shared_checkpoint(checkpoint, info):
    for name in ('pi', 'wm', 'critic'):
        if name not in checkpoint:
            continue  # Evaluation snapshots omit critic.
        state = {k.removeprefix('map_encoder.'): v for k, v in checkpoint[name].items()
                 if k.startswith('map_encoder.')}
        if encoder_hash(state) != info['sha256']:
            raise ValueError(f'{name} shared encoder hash differs from frozen AE metadata')


def encoder_from_checkpoint(checkpoint, device='cpu'):
    info = checkpoint.get('heightmap', {}).get('encoder')
    if info is None:
        return None, None
    validate_shared_checkpoint(checkpoint, info)
    encoder = HeightEncoder(info['map_dim'])
    encoder.load_state_dict({k.removeprefix('map_encoder.'): v for k, v in checkpoint['pi'].items()
                            if k.startswith('map_encoder.')})
    if encoder_info(encoder) != info:
        raise ValueError('Checkpoint encoder metadata is incompatible')
    return freeze_encoder(encoder.to(device)), info
