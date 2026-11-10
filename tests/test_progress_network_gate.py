"""A successful mean cannot admit a tunneled or long-tailed retest path."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('network_preflight',Path(__file__).resolve().parents[1]/'scripts/preflight_v6_progress_network.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NetworkGateTests(unittest.TestCase):
    def probe(self):
        return dict(complete=True,pairs=[{} for _ in range(20)],summary=dict(
            model_round_trip_ms=dict(p95=350,max=700),echo_ms=dict(p95=150,max=300)))

    def test_route_change_or_tunnel_blocks_even_good_latency(self):
        p=self.probe()
        self.assertTrue(module.network_gate(p,'en0','en0'))
        for before,after in [('en0','en1'),('utun4','utun4'),('en0','utun4')]:
            self.assertFalse(module.network_gate(p,before,after))

    def test_slow_tail_and_incomplete_samples_block(self):
        p=self.probe()
        for kind,field,value in [('model_round_trip_ms','max',1501),('model_round_trip_ms','p95',1001),
                                 ('echo_ms','max',1001),('echo_ms','p95',501)]:
            changed=copy.deepcopy(p);changed['summary'][kind][field]=value
            self.assertFalse(module.network_gate(changed,'en0','en0'))
        p['pairs'].pop();self.assertFalse(module.network_gate(p,'en0','en0'))
        p=self.probe();p['complete']=False;self.assertFalse(module.network_gate(p,'en0','en0'))
        self.assertFalse(module.network_gate({},'en0','en0'))


if __name__=='__main__':unittest.main()
