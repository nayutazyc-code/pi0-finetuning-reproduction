"""Train-only normalization compatibility aggregates; keep raw statistics on cloud."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
import pyarrow.parquet as pq


def summarize(values,stats):
    mean=np.asarray(stats['mean'],float)[:14];std=np.asarray(stats['std'],float)[:14]
    z=np.abs((values-mean)/(std+1e-6))
    return {'abs_normalized_max':float(z.max()),'abs_normalized_p99':float(np.quantile(z,.99)),
            'fraction_above10':float(np.mean(z>10))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();settings=tomllib.loads(args.config.read_text())
    sys.path.insert(0,str(root/'openpi/src'))
    from openpi.policies.aloha_policy import _encode_actions_inv,_decode_state
    repo=root/settings['data']['directory']/settings['data']['train_repo_id']
    mapping=json.loads((repo/'source_episodes.json').read_text())
    if mapping['source_split']!='train' or any(r['split']!='train' for r in mapping['episodes']):raise ValueError('Non-training mapping')
    states=[];actions=[]
    for path in sorted(repo.glob('data/chunk-*/episode_*.parquet')):
        table=pq.read_table(path,columns=['observation.state','action'])
        states.append(np.asarray(table['observation.state'].to_pylist(),np.float32))
        actions.append(np.asarray(table['action'].to_pylist(),np.float32))
    states=np.concatenate(states);actions=np.concatenate(actions)
    assert len(states)==sum(r['frames'] for r in mapping['episodes'])
    norm=root/settings['paths']['initial_checkpoint']/'assets/local/pi0_banana_train/norm_stats.json'
    stats=json.loads(norm.read_text())['norm_stats']
    actions=_encode_actions_inv(actions,adapt_to_pi=True)
    states=np.stack([_decode_state(row,adapt_to_pi=True) for row in states])
    result={'scope':'Aggregate compatibility screen only; allrawnormstats/values remain on cloud, not a distribution-quality guarantee.',
            'training_frames':len(states),'norm_stats_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),
            'actions':summarize(actions,stats['actions']),'states':summarize(states,stats['state'])}
    passed=all(np.isfinite(r['abs_normalized_max']) and r['abs_normalized_max']<50 and r['fraction_above10']<.01 for r in (result['actions'],result['states']))
    result['status']='passed' if passed else 'failed'
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not passed:raise ValueError('New data exceeds conservative normalization compatibility limits; inspect before training')


if __name__=='__main__':main()
