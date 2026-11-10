"""Inject an explicit training-only phase sampler into the official loader in this process."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import jax
from openpi.training import data_loader as dl


def install_phase_sampler(root, settings):
    record=json.loads((root/settings["sampling"]["weights_file"]).read_text())
    if record["split"]!="train" or record["repo_id"]!=settings["data"]["train_repo_id"]:
        raise ValueError("Sampler must be training-only")
    if record.get("dataset_manifest_sha256"):
        manifest=root/settings["data"]["directory"]/"dataset_manifest.json"
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=record["dataset_manifest_sha256"]:
            raise ValueError("Sampling weights belong to a different dataset manifest")
    weights=np.array(record["weights"],np.float64)
    if not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("Invalid sampling weights")
    def create(data_config, model_config, action_horizon, batch_size, *, sharding=None,
               skip_norm_stats=False, shuffle=False, num_batches=None, num_workers=0, seed=0, framework="jax"):
        if framework!="jax" or jax.process_count()!=1 or data_config.repo_id!=record["repo_id"]:
            raise ValueError("Phase sampler supports this single-GPU pi0 training dataset only")
        dataset=dl.create_torch_dataset(data_config,action_horizon,model_config)
        dataset=dl.transform_dataset(dataset,data_config,skip_norm_stats=skip_norm_stats)
        temporal = settings['model'].get('temporal_training', False)
        if temporal:
            from banana_temporal_data import EpisodeMaskedDataset
            mapping=json.loads((root/settings['data']['directory']/record['repo_id']/'source_episodes.json').read_text())
            if mapping['source_split'] != 'train' or any(r['split'] != 'train' or r['lerobot_episode_index'] != i for i,r in enumerate(mapping['episodes'])):
                raise ValueError('Invalid contiguous training episode mapping')
            dataset=EpisodeMaskedDataset(dataset,[r['frames'] for r in mapping['episodes']],action_horizon)
        if len(dataset)!=len(weights): raise ValueError("Sampler/dataset frame count mismatch")
        generator=torch.Generator().manual_seed(seed)
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(dataset),replacement=True,generator=generator)
        loader=dl.TorchDataLoader(dataset,local_batch_size=batch_size,sharding=sharding,
            shuffle=False,sampler=sampler,num_batches=num_batches,num_workers=num_workers,seed=seed,framework=framework)
        if temporal:
            from banana_temporal_data import TemporalDataLoader
            return TemporalDataLoader(data_config,loader)
        return dl.DataLoaderImpl(data_config,loader)
    dl.create_torch_data_loader=create
