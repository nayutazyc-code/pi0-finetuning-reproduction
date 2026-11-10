"""Before stopping a policy, reject mismatched source/data/norm or disk budget."""
import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('ik_job',Path(__file__).resolve().parents[1]/'outputs/ik_continuous_20261029/cloud_training_job.py')
job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)


class IKGuardTests(unittest.TestCase):
    def test_pinned_upstream_schedule_saves_only_final999(self):
        path=Path(__file__).resolve().parents[1]/'repos/openpi/scripts/train.py'
        tree=ast.parse(path.read_text())
        matches=[n for n in ast.walk(tree) if isinstance(n,ast.If) and any(
            isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='save_state'
            for statement in n.body for c in ast.walk(statement))]
        self.assertEqual(len(matches),1)
        predicate=compile(ast.Expression(matches[0].test),str(path),'eval')
        config=SimpleNamespace(save_interval=1000,num_train_steps=1000)
        saved=[step for step in range(1000) if eval(predicate,{},dict(config=config,step=step,start_step=0))]
        self.assertEqual(saved,[999])

    def fixture(self,root):
        def put(name,data):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data);return p
        source={}
        for n in job.REQUIRED_SOURCES:source[n]=job.sha(put(n,'test fixture'))
        norm=put(job.INITIAL+'/assets/local/pi0_banana_train/norm_stats.json','{"fixture":1}')
        put(job.INITIAL+'/_CHECKPOINT_METADATA','committed');put(job.INITIAL+'/params/_METADATA','params')
        volumes={k:root/('volume_'+k) for k in ('control','candidate')};arms=[]
        for label in volumes:
            ds='data/lerobot/'+label;meta={'source_split':'train','episodes':[{'split':'train','lerobot_episode_index':0,'frames':1}]}
            put(ds+'/local/pi0_banana_train/source_episodes.json',json.dumps(meta));manifest=put(ds+'/dataset_manifest.json','{"status":"complete"}')
            parquet=put(ds+'/local/pi0_banana_train/data/chunk-000/episode_000000.parquet','opaque fixture bytes')
            for split in ('validation','test'):put(ds+f'/local/pi0_banana_{split}/source_episodes.json',json.dumps({'episodes':[{'layout_id':split,'source_trajectory_sha256':'heldout','frames':1}]}))
            sampler=put('assets/'+label+'.json',json.dumps({'split':'train','repo_id':'local/pi0_banana_train','weights':[1.]}))
            config='\n'.join(['[data]',f'directory = "{ds}"','[model]','family = "pi0"','config_name = "pi0_banana_lora"','action_horizon = 100','strict_lora_only = true','train_action_projections = true','weighted_flow_loss = true','camera_crop = [275,110,540,345]','image_augmentation = false','[train]','batch_size = 8','gradient_accumulation_steps = 2','num_train_steps = 1000','save_interval = 1000','keep_period = 1000000','require_initial_norm_match = true','[paths]',f'initial_checkpoint = "{job.INITIAL}"',f'checkpoint_base_dir = "{volumes[label]}"','assets_base_dir = "assets_ik_continuous_norm"','[sampling]',f'weights_file = "assets/{label}.json"'])
            cfg=put('configs/'+label+'.toml',config)
            arms.append({'arm':label,'experiment':f'banana_v6_ik_{label}1000','config':str(cfg.relative_to(root)),'config_sha256':job.sha(cfg),'dataset':ds,'dataset_manifest_sha256':job.sha(manifest),'sampler':str(sampler.relative_to(root)),'sampler_sha256':job.sha(sampler),'train_episodes':1,'train_frames':1,'dataset_files_sha256':{str(parquet.relative_to(root)):job.sha(parquet)}})
        upstream=put('openpi/scripts/train.py','scheduler fixture')
        return {'upstream_train_sha256':job.sha(upstream),'schema':'v6-ik-continuous-training-train-only-v1','initial_checkpoint':job.INITIAL,'norm_stats_sha256':job.sha(norm),'source_sha256':source,'arms':arms,
                'heldout_lineage_local_verification':{'passed':True,'validation':{'base_digest':'same','candidate_digest':'same'},'test':{'base_digest':'same','candidate_digest':'same'}}},norm,volumes

    def test_good_budget_and_fixed_semantics_pass_without_writing_checkpoints(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();plan,norm,volumes=self.fixture(root)
            with patch.object(job,'NORM_SHA',job.sha(norm)),patch.object(job,'VOLUMES',volumes),patch.object(job.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)):
                settings,_,report=job.validate(root,plan)
            self.assertTrue(report['heldout_lineage_locally_verified']);self.assertFalse(report['heldout_contents_uploaded']);self.assertEqual(len(settings),2)
            self.assertTrue(all(not p.exists() for p in volumes.values()))

    def test_bad_norm_missing_source_and_disk_shortage_stop_preflight(self):
        for mode in ('norm','source','disk','unbound_parquet'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                root=Path(t).resolve();plan,norm,volumes=self.fixture(root);digest=job.sha(norm)
                if mode=='norm':norm.write_text('changed')
                if mode=='source':plan['source_sha256'].pop('server/banana_finetune.py')
                if mode=='unbound_parquet':plan['arms'][0]['dataset_files_sha256']={}
                with patch.object(job,'NORM_SHA',digest),patch.object(job,'VOLUMES',volumes),patch.object(job.shutil,'disk_usage',return_value=SimpleNamespace(free=1 if mode=='disk' else 10*1024**3)):
                    with self.assertRaises(ValueError):job.validate(root,plan)


if __name__=='__main__':unittest.main()
