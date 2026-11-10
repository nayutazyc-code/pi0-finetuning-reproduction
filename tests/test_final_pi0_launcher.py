"""Exercise the final runtime lifecycle without SSH, GPU or physics."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('final_launcher',ROOT/'scripts/run_final_pi0.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)

class FinalLauncherTest(unittest.TestCase):
    def test_final_profile_values_and_scene_are_exact(self):
        a=tomllib.loads((ROOT/'configs/banana_final_pi0.toml').read_text())
        b=tomllib.loads((ROOT/'configs/banana_ik_continuous_damping2_pi0.toml').read_text())
        self.assertEqual(a,b)
        cfg,scene=launcher.local_config()
        self.assertEqual(cfg.model.expected_checkpoint_suffix,launcher.CANDIDATE)
        self.assertEqual(scene.wrist_damping_scale,2)
        self.assertEqual(cfg.simulation.seed,20261027)
        self.assertEqual(cfg.simulation.episodes,1)
        self.assertFalse(cfg.simulation.display)
        self.assertTrue(cfg.simulation.save_video)
        cfg,scene=launcher.local_config(True)
        self.assertEqual(cfg.model.expected_checkpoint_suffix,launcher.V6)
        self.assertEqual(scene.wrist_damping_scale,1)

    def test_check_only_never_constructs_session_or_calls_remote(self):
        with patch.object(launcher,'Session') as session,patch.object(launcher.subprocess,'run') as remote,patch.object(launcher.subprocess,'Popen') as popen,redirect_stdout(io.StringIO()) as output:
            launcher.main(['--check-only'])
        self.assertTrue(json.loads(output.getvalue())['valid'])
        session.assert_not_called();remote.assert_not_called();popen.assert_not_called()

    def test_v6_fallback_does_not_depend_on_candidate_files(self):
        original_loader=launcher.load_config
        def loader(path):
            if path.name!='banana_v6_bridge20_pi0.toml':raise ValueError('Candidate configuration damaged')
            return original_loader(path)
        with patch.object(launcher,'load_config',side_effect=loader):
            cfg,scene=launcher.local_config(True)
        self.assertEqual(cfg.model.expected_checkpoint_suffix,launcher.V6)
        self.assertEqual(scene.wrist_damping_scale,1)

    def lifecycle(self,*,select_only=False,fallback=False,dispatch_error=False,metadata_error=False,run_error=False,interrupt=False,fallback_error=False):
        cfg,scene=launcher.local_config(fallback)
        session=launcher.Session(cfg,scene,fallback=fallback,select_only=select_only)
        events=[]
        def setup():
            session.out=Path(directory);session.target='private-target';session.record={}
        def remote(args,**kwargs):
            model=args[-1];events.append(model)
            if model=='candidate' and dispatch_error:raise RuntimeError('Lost dispatch response')
            return dict(model=model)
        def endpoint(config):
            if config.model.expected_checkpoint_suffix==launcher.CANDIDATE and metadata_error:raise ValueError('Checkpoint mismatch')
            if config.model.expected_checkpoint_suffix==launcher.V6 and fallback_error:raise RuntimeError('Fallback offline')
            return dict(checkpoint=config.model.expected_checkpoint_suffix)
        simulate=MagicMock(return_value=Path('/private/tmp/mock-video'))
        if run_error:simulate.side_effect=RuntimeError('Simulation failed')
        if interrupt:simulate.side_effect=KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(session,'setup',side_effect=setup),patch.object(session,'ensure_proxy'),patch.object(session,'start_tunnel'),patch.object(session,'remote',side_effect=remote),patch.object(session,'endpoint',side_effect=endpoint),patch.object(session,'simulate',simulate),patch.object(session,'close') as close,redirect_stdout(io.StringIO()):
                if dispatch_error or metadata_error or run_error or interrupt:
                    with self.assertRaises((RuntimeError,ValueError,KeyboardInterrupt)):session.execute()
                else:session.execute()
                close.assert_called_once()
            record=json.loads((Path(directory)/'completion.json').read_text())
        return events,record,simulate

    def test_success_keeps_candidate(self):
        events,record,simulate=self.lifecycle()
        self.assertEqual(events,['candidate']);self.assertTrue(record['success']);self.assertTrue(record['kept_selected_service'])
        simulate.assert_called_once()

    def test_select_only_never_steps(self):
        events,record,simulate=self.lifecycle(select_only=True)
        self.assertEqual(events,['candidate']);self.assertTrue(record['success']);simulate.assert_not_called()

    def test_explicit_fallback_only_selects_v6(self):
        events,record,simulate=self.lifecycle(fallback=True)
        self.assertEqual(events,['v6']);self.assertTrue(record['success']);simulate.assert_not_called()

    def test_dispatch_lost_attempts_real_restore_path(self):
        events,record,simulate=self.lifecycle(dispatch_error=True)
        self.assertEqual(events,['candidate','v6']);self.assertTrue(record['fallback_verified']);simulate.assert_not_called()

    def test_metadata_mismatch_attempts_real_restore_path(self):
        events,record,simulate=self.lifecycle(metadata_error=True)
        self.assertEqual(events,['candidate','v6']);self.assertTrue(record['fallback_verified']);simulate.assert_not_called()

    def test_run_error_attempts_real_restore_path(self):
        events,record,_=self.lifecycle(run_error=True)
        self.assertEqual(events,['candidate','v6']);self.assertTrue(record['fallback_verified'])

    def test_keyboard_interrupt_attempts_real_restore_path(self):
        events,record,_=self.lifecycle(interrupt=True)
        self.assertEqual(events,['candidate','v6']);self.assertEqual(record['error_type'],'KeyboardInterrupt');self.assertTrue(record['fallback_verified'])

    def test_restore_error_does_not_mask_original(self):
        events,record,_=self.lifecycle(run_error=True,fallback_error=True)
        self.assertEqual(events,['candidate','v6']);self.assertEqual(record['error'],'Simulation failed')
        self.assertFalse(record['fallback_verified']);self.assertEqual(record['fallback_error'],'Fallback offline')

    def proxy_branch(self,existing):
        cfg,scene=launcher.local_config();session=launcher.Session(cfg,scene)
        sock=MagicMock();sock.connect_ex.return_value=0 if existing else 1
        popen=MagicMock();popen.return_value.poll.return_value=None;popen.return_value.pid=1234
        def remote(args,**kwargs):
            # Execute the production remote startup code with fake OS surfaces.
            code=args[3]
            fake_socket=SimpleNamespace(socket=MagicMock(return_value=sock))
            fake_subprocess=SimpleNamespace(Popen=popen,DEVNULL=-3,STDOUT=-2)
            fake_pathlib=SimpleNamespace(Path=lambda _:directory_path)
            with patch.dict(sys.modules,{'socket':fake_socket,'subprocess':fake_subprocess,'time':SimpleNamespace(sleep=lambda _:None),'pathlib':fake_pathlib}),patch.object(sys,'argv',['remote','unique']),redirect_stdout(io.StringIO()) as output:
                exec(compile(code,'remote_proxy','exec'),{})
            return json.loads(output.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            directory_path=Path(directory);session.out=directory_path
            with patch.object(session,'remote',side_effect=remote):session.ensure_proxy()
            receipt=json.loads((Path(directory)/'proxy.json').read_text())
        return receipt,popen

    def test_cold_proxy_starts_owned_loopback_sidecar(self):
        receipt,popen=self.proxy_branch(False)
        self.assertTrue(receipt['started']);popen.assert_called_once()
        self.assertIn('127.0.0.1',popen.call_args.args[0]);popen.return_value.terminate.assert_not_called()

    def test_existing_proxy_is_preserved(self):
        receipt,popen=self.proxy_branch(True)
        self.assertTrue(receipt['existing']);self.assertFalse(receipt['started']);popen.assert_not_called()

    def test_endpoint_metadata_error_is_not_retried(self):
        cfg,scene=launcher.local_config();session=launcher.Session(cfg,scene)
        with patch.object(launcher,'RemotePi0',side_effect=ValueError('Wrong checkpoint')) as remote:
            with self.assertRaises(ValueError):session.endpoint(cfg)
        remote.assert_called_once()

if __name__=='__main__':unittest.main()
