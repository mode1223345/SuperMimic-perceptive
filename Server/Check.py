"""Assert smoke artifacts before starting expensive comparison runs."""
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from supermimic.Ae import load_encoder, validate_shared_checkpoint


def main():
    _, info = load_encoder(ROOT / 'logs/AeFinal/Encoder.pt')
    report = {'encoder': info, 'checks': {}}
    for name, expected_iteration in [('Smoke', 8), ('Compiled', 6)]:
        ck = torch.load(ROOT / f'logs/{name}/last.pt', map_location='cpu', weights_only=False)
        assert ck['iteration'] == expected_iteration
        assert ck['heightmap']['encoder'] == info
        validate_shared_checkpoint(ck, info)
        with (ROOT / f'logs/{name}/train_log.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert all(math.isfinite(float(value)) for row in rows for value in row.values())
        assert any(float(row['pi_gradnorm']) > 0 for row in rows)
        report['checks'][name] = {'iteration': ck['iteration'], 'encoder_unchanged': True,
                                 'all_metrics_finite': True, 'actor_received_gradient': True,
                                 'last_metrics': rows[-1]}
    evaluations = list((ROOT / 'logs/Smoke/Eval_full').glob('ckpt_*.json'))
    assert evaluations, 'Full-motion evaluation did not produce results'
    evaluated = json.loads(evaluations[-1].read_text())
    assert len(evaluated['per_clip']) == 40
    report['checks']['evaluation'] = {'motions': 40, 'protocol': 'full',
                                     'note': 'Untrained smoke policy; score is not a performance claim.'}
    path = ROOT / 'results/Validation.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
