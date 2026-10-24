"""Pretrain a geometry AE without launching the simulator."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.Ae import Autoencoder, reconstruction_loss, reconstruction_metrics, sample_maps, save_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='logs/Ae')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--train_maps', type=int, default=65536)
    parser.add_argument('--valid_maps', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--map_dim', type=int, default=32)
    parser.add_argument('--log_every', type=int, default=200)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.train_maps, args.valid_maps, args.map_dim, args.log_every) <= 0:
        parser.error('Counts must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'Args.json').write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    train = sample_maps(args.train_maps, args.seed, 'train', args.device)
    valid = sample_maps(args.valid_maps, args.seed, 'validation', args.device)
    model = Autoencoder(args.map_dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_step = float('inf'), 0
    history = []
    # Predicting the per-map mean height exposes a misleading near-flat baseline.
    naive = torch.zeros_like(valid)
    valid_count = valid[:, 1].sum((1, 2), keepdim=True).clamp_min(1)
    naive[:, 0] = (valid[:, 0] * valid[:, 1]).sum((1, 2), keepdim=True) / valid_count
    naive[:, 1] = (valid[:, 1] * 2 - 1) * 20
    naive_metrics = reconstruction_metrics(naive, valid)
    with (out / 'Metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'seconds', 'train_loss', 'valid_loss', 'height_rmse_m', 'edge_rmse_m', 'validity_accuracy'])
        for step in range(1, args.steps + 1):
            idx = torch.randint(len(train), (args.batch,), device=args.device)
            batch = train[idx]
            loss = reconstruction_loss(model(batch), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
            if step % args.log_every == 0 or step == args.steps:
                with torch.no_grad():
                    prediction = torch.cat([model(chunk) for chunk in valid.split(args.batch)])
                    val_loss = float(reconstruction_loss(prediction, valid))
                    metrics = reconstruction_metrics(prediction, valid)
                row = [step, time.time() - started, float(loss.detach()), val_loss, metrics['height_rmse_m'],
                       metrics['edge_rmse_m'], metrics['validity_accuracy']]
                writer.writerow(row)
                handle.flush()
                history.append(row)
                print(json.dumps(dict(zip(['step', 'seconds', 'train_loss', 'valid_loss', 'rmse_m', 'edge_rmse_m', 'mask_acc'], row))), flush=True)
                if val_loss < best:
                    best, best_step = val_loss, step
                    torch.save({'model': model.state_dict(), 'args': vars(args), 'step': step}, out / 'Best.pt')
    model.load_state_dict(torch.load(out / 'Best.pt', map_location=args.device, weights_only=False)['model'])
    with torch.no_grad():
        prediction = torch.cat([model(chunk) for chunk in valid.split(args.batch)])
        metrics = reconstruction_metrics(prediction, valid)
        profiles = {}
        for profile in ('flat', 'steps'):
            samples = sample_maps(min(args.valid_maps, 4096), args.seed + 17, 'validation', args.device, profile)
            output = torch.cat([model(chunk) for chunk in samples.split(args.batch)])
            profiles[profile] = reconstruction_metrics(output, samples)
    report = {'best_step': best_step, 'best_valid_loss': best, 'seconds': time.time() - started,
              'validation': metrics, 'canonical_profiles': profiles, 'per_map_mean_baseline': naive_metrics,
              'data': {'train_maps': len(train), 'valid_maps': len(valid),
                       'note': 'Disjoint procedural geometries; canonical Flat/Steps shared, with disjoint poses.'}}
    report['encoder'] = save_encoder(out / 'Encoder.pt', model.encoder, {**vars(args), **report})
    (out / 'Report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Inspect edge-rich held-out maps, not just easy flat examples.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    energy = ((valid[:, 0, :, 1:] - valid[:, 0, :, :-1]).abs() * valid[:, 1, :, 1:] * valid[:, 1, :, :-1]).sum((1, 2))
    indices = energy.argsort(descending=True)[::max(1, len(valid) // 12)][:6]
    fig, axes = plt.subplots(3, len(indices), figsize=(15, 7), constrained_layout=True, squeeze=False)
    for col, idx in enumerate(indices):
        truth, pred = valid[idx].cpu(), prediction[idx].cpu()
        for row, img in enumerate((truth[0], pred[0], (pred[0] - truth[0]).abs() * truth[1])):
            options = dict(vmin=float(truth[0][truth[1].bool()].min()), vmax=float(truth[0][truth[1].bool()].max())) if row < 2 and truth[1].any() else {}
            axes[row, col].imshow(img, origin='lower', cmap='viridis' if row < 2 else 'magma', **options)
            axes[row, col].set_title(('Truth', 'Reconstruction', 'Absolute error')[row])
            axes[row, col].axis('off')
    fig.savefig(out / 'Reconstruction.png', dpi=140)
    plt.close(fig)
    print('PRETRAIN_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
