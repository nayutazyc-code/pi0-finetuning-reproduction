"""Deadline recovery, default equivalence and opt-in timing lifecycle."""
import gc
from pathlib import Path
import random
import tempfile
import time
import unittest

from pi0_repro.config import PROJECT_ROOT, load_config
from pi0_repro.pacing import next_apply_deadline
from pi0_repro.control_timing import ControlTimingTrace, measured_call


class PacingTests(unittest.TestCase):
    def test_default_exactly_matches_legacy_clock(self):
        rng=random.Random(20261029)
        for _ in range(10000):
            scheduled=rng.uniform(0,100);applied=scheduled+rng.uniform(0,.3)
            finished=applied+rng.uniform(0,.1);late=max(0.,applied-scheduled)
            expected=applied+.02 if late>.002 else scheduled+.02
            if finished>=expected:expected=finished+.02
            self.assertEqual(next_apply_deadline(scheduled,applied,finished,late),expected)

    def test_overrun_removes_extra_wait_without_immediate_command_burst(self):
        for duration in (.025,.035,.05,.08):
            old=next_apply_deadline(1.,1.,1.+duration,0.)
            new=next_apply_deadline(1.,1.,1.+duration,0.,recover_overrun=True)
            self.assertAlmostEqual(old-new,.02)
            self.assertGreaterEqual(new,1.+.02)
            self.assertEqual(new,1.+duration)
        # Finish after the anchored deadline but before last application +20ms.
        self.assertEqual(next_apply_deadline(1.,1.001,1.0205,.001,recover_overrun=True),1.021)
        # Ordinary steps preserve the old anchored clock exactly.
        self.assertEqual(next_apply_deadline(1.,1.001,1.005,.001,recover_overrun=True),1.02)

    def test_flags_are_opt_in_typed_and_require_appropriate_clock(self):
        source=(PROJECT_ROOT/'configs/banana_v6_bridge20_pi0.toml').read_text()
        cfg=load_config(PROJECT_ROOT/'configs/banana_v6_bridge20_pi0.toml')
        self.assertFalse(cfg.simulation.recover_control_overruns);self.assertFalse(cfg.simulation.trace_control_timing)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.toml'
            p.write_text(source+'\nrecover_control_overruns = true\ntrace_control_timing = true\n')
            self.assertTrue(load_config(p).simulation.recover_control_overruns)
            for bad in (source+'\nrecover_control_overruns = "yes"\n',
                        source+'\ntrace_control_timing = "yes"\n',
                        source.replace('control_pacing = "before_step"','control_pacing = "after_step"')+'\nrecover_control_overruns = true\n',
                        source.replace('pi0_repro/BananaBox-v0','gym_aloha/AlohaTransferCube-v0')+'\ntrace_control_timing = true\n'):
                p.write_text(bad)
                with self.assertRaises(ValueError):load_config(p)

    def test_trace_restores_gc_hooks_and_phase_records_on_error(self):
        original=list(gc.callbacks);trace=ControlTimingTrace(time.perf_counter());trace.step=7
        try:
            gc.collect()
            self.assertTrue(trace.events)
            self.assertTrue(all(e['collector_is_control_thread'] and e['start_control_step']==7 for e in trace.events))
            timings={}
            def bad():raise RuntimeError('phase failure')
            with self.assertRaisesRegex(RuntimeError,'phase failure'):measured_call(timings,'phase',bad)
            self.assertGreaterEqual(timings['phase']['wall_seconds'],0.)
        finally:receipt=trace.close()
        self.assertEqual(gc.callbacks,original);self.assertEqual(receipt['unfinished_gc_callbacks'],0)
        trace.close();self.assertEqual(gc.callbacks,original)


if __name__=='__main__':unittest.main()
