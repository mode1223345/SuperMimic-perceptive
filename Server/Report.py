"""Read-only experiment metrics to TensorBoard and a comparison figure."""
import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
RUNS = ('FlatAe', 'StepsAe', 'FlatBase', 'StepsBase')
TAGS = {'wm_loss': 'WorldModel/loss', 'wm_err8': 'WorldModel/error8', 'pi_loss': 'Policy/loss',
        'v_loss': 'Critic/loss', 'ep_ret_norm': 'Train/episode_return', 'sim_rew': 'Train/step_reward',
        'imag_rew': 'Imagination/reward', 'imag_survival': 'Imagination/survival',
        'pi_gradnorm': 'Policy/gradient_norm', 'it_per_s': 'Performance/iterations_per_second',
        'sim_steps': 'Data/transitions', 'wall_h': 'Time/training_hours'}
ETAGS = {'norm_return': 'Eval/normalized_return', 'norm_return_framew': 'Eval/frame_weighted_return',
         'n_failed': 'Eval/failed_clips'}


def rows(path):
    if not path.exists():
        return []
    result = []
    with path.open() as handle:
        for record in csv.DictReader(handle):
            try:
                values = {k: float(v) for k, v in record.items() if k not in ('failed', 'saved_at', 'eval_at')}
                if all(math.isfinite(v) for v in values.values()):
                    result.append(values)
            except (ValueError, TypeError):
                pass  # Concurrent CSV write can leave an incomplete last row.
    return result


def main():
    OUT.mkdir(exist_ok=True)
    writers = {name: SummaryWriter(str(ROOT / 'logs/Tensorboard' / name), purge_step=0) for name in RUNS}
    seen = {name: {'train': -1, 'eval': -1} for name in RUNS}
    ae = json.loads((ROOT / 'logs/AeFinal/Report.json').read_text())
    with SummaryWriter(str(ROOT / 'logs/Tensorboard/Pretrain'), purge_step=0) as writer:
        for record in rows(ROOT / 'logs/AeFinal/Metrics.csv'):
            for key in ('train_loss', 'valid_loss', 'height_rmse_m', 'edge_rmse_m', 'validity_accuracy'):
                writer.add_scalar('AE/' + key, record[key], int(record['step']))
    for name, writer in writers.items():
        writer.add_text('Protocol', 'Fresh seed 0; 4096 env/pi/WM; full 40-motion evaluation. AE pretraining cost separate. Single seed; no statistical claim.', 0)
        writer.add_scalar('Time/pretrain_seconds', ae['seconds'] if name.endswith('Ae') else 0, 0)
    while True:
        state, data = {}, {}
        for name, writer in writers.items():
            train = rows(ROOT / 'logs' / name / 'train_log.csv')
            ev = rows(ROOT / 'logs' / name / 'Eval_full/eval.csv')
            data[name] = train, ev
            for kind, records, tags in (('train', train, TAGS), ('eval', ev, ETAGS)):
                for record in records:
                    step = int(record['iteration'])
                    if step <= seen[name][kind]:
                        continue
                    for key, tag in tags.items():
                        writer.add_scalar(tag, record[key], step)
                    seen[name][kind] = step
            writer.flush()
            state[name] = {'train': train[-1] if train else None, 'eval': ev[-1] if ev else None}
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, (kind, key, title) in zip(axes.flat, [(1, 'norm_return', 'Full-motion evaluation return'),
                (1, 'n_failed', 'Failed motions / 40'), (0, 'ep_ret_norm', 'Training episode return'),
                (0, 'wm_loss', 'Dynamics loss'), (0, 'v_loss', 'Value loss'), (0, 'it_per_s', 'Training iterations / second')]):
            for name in RUNS:
                record = data[name][kind]
                ax.plot([r['iteration'] for r in record], [r[key] for r in record], label=name,
                        color='#245b91' if name.startswith('Flat') else '#ad591e',
                        linestyle='-' if name.endswith('Ae') else '--')
            ax.set(title=title, xlabel='Iteration')
            ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle('Frozen shared AE (solid) vs independent trainable CNNs (dashed) | seed 0')
        fig.savefig(OUT / 'Curves.tmp.png', dpi=140)
        (OUT / 'Curves.tmp.png').replace(OUT / 'Curves.png')
        plt.close(fig)
        (OUT / 'Status.tmp.json').write_text(json.dumps(state, indent=2))
        (OUT / 'Status.tmp.json').replace(OUT / 'Status.json')
        time.sleep(30)


if __name__ == '__main__':
    main()
