"""Plot every full policy trajectory; no shifted or favorable time windows."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/release_retreat_20261029/closed_loop_replication'
ARMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / 'outputs/release_retreat_20261029')
    read = lambda p: json.loads(p.read_text())
    summary, audit = read(out / 'aggregate.json'), read(out / 'audit.json')
    assert audit['passed'] and summary['complete'] and len(summary['trials']) == 24
    dest = out / 'plots'
    dest.mkdir(exist_ok=False)
    colors = dict(v6='#b34d24', v6_damping2='#366bb0', candidate='#228359', candidate_damping2='#8e53a5')
    plots = []
    for metric in ('jerk', 'wrist_angle'):
        fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, constrained_layout=True)
        for ax, layout in zip(axes.flat, sorted({t['layout'] for t in summary['trials']}), strict=True):
            for trial in (t for t in summary['trials'] if t['layout'] == layout):
                path = Path(trial['run_directory']) / 'episode_000.npz'
                with np.load(path, allow_pickle=False) as data:
                    q = data['state'].astype(float)
                if metric == 'jerk':
                    values = np.max(abs(np.diff(q[:, ARMS], n=3, axis=0)), axis=1) / .02**3
                    assert np.isclose(values.max(), trial['quality']['all_frame_max_jerk_rad_s3']['max'], rtol=1e-12)
                    times = np.arange(3, len(q)) * .02
                else:
                    values, times = q[:, 11], np.arange(len(q)) * .02
                label = f"{trial['model']} r{trial['repeat']} {'OK' if trial['success'] else 'FAIL'}"
                ax.plot(times, values, color=colors[trial['model']], ls='-' if trial['repeat'] == 1 else '--',
                        lw=.85, alpha=.85, label=label)
            ax.set_title(layout)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, ncol=2)
            ax.set_ylabel('max arm |jerk| (rad/s³)' if metric == 'jerk' else 'right wrist angle (rad)')
            if metric == 'jerk':
                ax.set_yscale('symlog', linthresh=1)
        for ax in axes[-1]:
            ax.set_xlabel('Simulation seconds from reset; full recorded trajectory')
        fig.suptitle('Actual pi0 policy | six validation layouts, one attempt each\n'
                     'RTC20; declared wrist damping only; unpaired noise; failures retained', fontsize=14)
        path = dest / (metric + '_all24.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(dict(path=str(path), sha256=sha(path)))
    manifest = dict(schema='ik-policy-full-trajectory-plots-v1', script_sha256=sha(Path(__file__)),
                    audit_sha256=sha(out / 'audit.json'), aggregate_sha256=sha(out / 'aggregate.json'),
                    plots=plots, visual_review_pending=True,
                    scope='All24 independent replication traces, simulation-time derivatives. Full reset-to-termination recordings; no clipping to favorable phases, temporal averaging, or policy-noise pairing. Wall-clock waits are reported separately in aggregate.')
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('COMPLETE', dest)


if __name__ == '__main__':
    main()
