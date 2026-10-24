"""Static terrain and differentiable, gravity-aligned local height queries.

The simulator uses TerrainSpec.boxes; the map rasterizes the same surfaces.
Bilinear interpolation is a continuous surrogate near the cuboid boundaries,
not a differentiable contact solver. Heights use metres and terrain_z - base_z.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TerrainSpec:
    """Immutable geometry shared by the simulator, replay and map queries.

    bounds is (xmin, xmax, ymin, ymax), at cell-aligned world coordinates.
    Each box is (cx, cy, size_x, size_y, top_z), with its bottom at z=0.
    The infinite flat ground is at z=0; bounds restrict map availability only.
    """

    profile: str = "flat"
    cell_size: float = 0.05
    bounds: tuple = (-10.0, 10.0, -10.0, 10.0)
    boxes: tuple | None = None

    def __post_init__(self):
        if self.profile not in ("flat", "steps"):
            raise ValueError("Terrain profile must be 'flat' or 'steps'")
        cell = float(self.cell_size)
        if not math.isfinite(cell) or cell <= 0:
            raise ValueError("cell_size must be finite and positive")
        bounds = tuple(float(v) for v in self.bounds)
        if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
            raise ValueError("bounds must contain four finite coordinates")
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError("Terrain bounds must have positive extent")
        if any(abs(v / cell - round(v / cell)) > 1e-6 for v in bounds):
            raise ValueError("Terrain bounds must align with cell_size")
        boxes = self.boxes
        if boxes is None:
            boxes = () if self.profile == "flat" else (
                (1.25, 0.0, 0.5, 1.2, 0.04),
                (2.0, 0.0, 0.5, 1.2, 0.08),
                (2.75, 0.0, 0.5, 1.2, 0.04),
            )
        boxes = tuple(tuple(float(v) for v in box) for box in boxes)
        if self.profile == "flat" and boxes:
            raise ValueError("Flat terrain cannot contain raised boxes")
        for box in boxes:
            if len(box) != 5 or not all(math.isfinite(v) for v in box):
                raise ValueError("Each box must contain five finite values")
            if box[2] <= 0 or box[3] <= 0 or box[4] <= 0:
                raise ValueError("Box dimensions and height must be positive")
        object.__setattr__(self, "cell_size", cell)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "boxes", boxes)

    def to_dict(self):
        return {"profile": self.profile, "cell_size": self.cell_size,
                "bounds": list(self.bounds), "boxes": [list(b) for b in self.boxes]}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @property
    def fingerprint(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def surface_height(self, xy: torch.Tensor) -> torch.Tensor:
        """Exact upper surface for reset checks; not the differentiable query."""
        if xy.shape[-1] != 2:
            raise ValueError("Surface coordinates must end with dimension 2")
        if not xy.is_floating_point():
            xy = xy.float()
        z = torch.zeros_like(xy[..., 0])
        for cx, cy, sx, sy, top in self.boxes:
            inside = ((xy[..., 0] - cx).abs() <= sx / 2) & ((xy[..., 1] - cy).abs() <= sy / 2)
            z = torch.where(inside, torch.maximum(z, z.new_tensor(top)), z)
        return z


class Heightmap(nn.Module):
    """Query [B, 2, 15, 23] local maps without replicating the global raster.

    Channel 0 is relative height, channel 1 is validity (0/1). Rows increase
    local y from -.35 to +.35 m; columns increase local x from -.55 to +.55 m.
    p and R are base pose in the same world frame used by TerrainSpec.
    Only yaw rotates the grid. Gradients reach p and R through grid_sample.
    No height clipping is applied. Out-of-bounds/unknown samples are (0, 0).
    """

    def __init__(self, spec: TerrainSpec, device="cpu"):
        super().__init__()
        self.spec = spec
        xmin, xmax, ymin, ymax = spec.bounds
        nx = round((xmax - xmin) / spec.cell_size) + 1
        ny = round((ymax - ymin) / spec.cell_size) + 1
        x = torch.linspace(xmin, xmax, nx, device=device)
        y = torch.linspace(ymin, ymax, ny, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        heights = spec.surface_height(torch.stack((xx, yy), dim=-1))
        # One shared image; both channels are sampled in a single operation.
        self.register_buffer("raster", torch.stack((heights, torch.ones_like(heights)))[None])
        ly, lx = torch.meshgrid(torch.linspace(-0.35, 0.35, 15, device=device),
                                torch.linspace(-0.55, 0.55, 23, device=device), indexing="ij")
        self.register_buffer("local_xy", torch.stack((lx, ly), dim=-1))

    @property
    def heights(self):
        return self.raster[:, :1]

    @property
    def validity(self):
        return self.raster[:, 1:2]

    def forward(self, p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        if p.ndim != 2 or p.shape[-1] != 3 or R.shape != (p.shape[0], 3, 3):
            raise ValueError("Expected base position [B,3] and rotation [B,3,3]")
        batch = p.shape[0]
        if batch == 0:
            return self.raster.new_empty((0, 2, 15, 23))
        # Keep the geometry query in the map dtype under mixed precision.
        p = p.to(dtype=self.raster.dtype)
        R = R.to(dtype=self.raster.dtype)
        heading = R[:, :2, 0]
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c, s = heading[:, 0, None, None], heading[:, 1, None, None]
        lx, ly = self.local_xy[..., 0], self.local_xy[..., 1]
        x = p[:, 0, None, None] + c * lx - s * ly
        y = p[:, 1, None, None] + s * lx + c * ly
        xmin, xmax, ymin, ymax = self.spec.bounds
        grid = torch.stack((2 * (x - xmin) / (xmax - xmin) - 1,
                            2 * (y - ymin) / (ymax - ymin) - 1), dim=-1)
        # B query windows are concatenated along output rows, not input images.
        sampled = F.grid_sample(self.raster, grid.reshape(1, batch * 15, 23, 2),
                                mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.reshape(2, batch, 15, 23).permute(1, 0, 2, 3)
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        # Require the interpolation footprint to be known; do not average holes
        # into a plausible flat surface. The mask is intentionally discrete.
        valid = inside & (sampled[:, 1] >= 1 - 1e-6)
        relative = sampled[:, 0] - p[:, 2, None, None]
        relative = torch.where(valid, relative, torch.zeros_like(relative))
        return torch.stack((relative, valid.to(relative.dtype)), dim=1)


class HeightEncoder(nn.Module):
    """Small spatial encoder; flattening preserves where a terrain feature is."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.ELU(),
                                 nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ELU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 4 * 6, 64), nn.ELU(),
                                  nn.Linear(64, latent_dim), nn.ELU())

    def forward(self, heightmap):
        return self.head(self.cnn(heightmap))
