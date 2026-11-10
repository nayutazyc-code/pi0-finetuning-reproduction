"""CPU real-reader audit for explicit episode-tail masks and unchanged targets."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from banana_config import require_openpi_commit, make_config, read_settings
    root = args.root.resolve()
    settings = read_settings(args.config)
    checkout = require_openpi_commit(root)
    os.environ['HF_LEROBOT_HOME'] = str(root/settings['data']['directory'])
    sys.path.insert(0, str(checkout/'src'))
    from openpi.training import data_loader as dl
    from banana_temporal_data import EpisodeMaskedDataset, TemporalObservation
    from openpi.models.model import Observation
    import jax
    import jax.numpy as jnp
    config = make_config(root, args.config, exp_name='v8_reader_audit')
    data = config.data.create(config.assets_dirs, config.model)
    if data.use_quantile_norm:
        raise ValueError('Temporal native std conversion requires standard normalization')
    mapping = json.loads((root/settings['data']['directory']/settings['data']['train_repo_id']/'source_episodes.json').read_text())
    raw = dl.create_torch_dataset(data, config.model.action_horizon, config.model)
    transformed = dl.transform_dataset(raw, data)
    lengths = [r['frames'] for r in mapping['episodes']]
    masked = EpisodeMaskedDataset(transformed, lengths, config.model.action_horizon)
    rows = []
    start = 0
    for n in lengths:
        end = start+n
        for index in (start, end-2, end-1):
            original = raw[index]
            sample = masked[index]
            mask = sample['action_valid_mask']
            expected = np.arange(100) < end-index
            assert np.array_equal(mask, expected)
            assert np.array_equal(mask, ~np.asarray(original['action_is_pad']))
            assert np.array_equal(sample['actions'], transformed[index]['actions'])
            assert sample['actions'].shape == (100,32)
            batch = jax.tree.map(lambda value:np.asarray(value)[None],sample)
            base = Observation.from_dict(batch)
            obs = TemporalObservation(**vars(base), action_valid_mask=mask)
            leaves = jax.tree.leaves(obs)
            assert any(np.shape(v)==(100,) and np.asarray(v).dtype==bool for v in leaves)
            rows.append({'index':index,'valid_steps':int(mask.sum())})
        start=end
    # Verify the extended observation survives batching, JIT and tree slicing.
    obs = obs.replace(action_valid_mask=obs.action_valid_mask[None])
    obs = jax.tree.map(jnp.asarray, obs)
    assert int(jax.jit(lambda value:value.action_valid_mask.sum())(obs)) == 1
    result={'status':'passed','checked_samples':len(rows),'training_frames':len(masked),
            'all_masks_match_lerobot_action_is_pad':True,'actions_unchanged':True,
            'temporal_observation_jit_passed':True,'samples':rows}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}))


if __name__=='__main__':
    main()
