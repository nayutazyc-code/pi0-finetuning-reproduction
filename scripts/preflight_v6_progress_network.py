"""Require direct-route and bounded no-step latency before a separate pilot."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def network_gate(probe,before,after):
    s=probe.get('summary') or {}
    model=s.get('model_round_trip_ms') or {};echo=s.get('echo_ms') or {}
    return bool(before==after and not after.startswith(('utun','tun','tap')) and probe.get('complete')
                and len(probe.get('pairs',[]))==20 and model.get('p95',float('inf'))<=1000
                and model.get('max',float('inf'))<=1500 and echo.get('p95',float('inf'))<=500
                and echo.get('max',float('inf'))<=1000)


def main():
    parent=ROOT/'outputs/v6_progress_20261028'
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    # Only interface names are retained. No private host or address enters receipts.
    def interface():
        route=subprocess.check_output(['route','-n','get',target],text=True,timeout=15)
        match=re.search(r'interface:\s*(\S+)',route)
        assert match,'No route interface'
        return match.group(1)
    before=interface()
    assert not before.startswith(('utun','tun','tap')),'Virtual tunnel still active; do not run retest'
    output=parent/'network_retest_preflight'
    assert not output.exists(),'Preserve earlier preflight attempts; choose an explicit amendment instead'
    gates=dict(model_round_trip_p95_ms_max=1000,model_round_trip_max_ms_max=1500,
               echo_p95_ms_max=500,echo_max_ms_max=1000,pairs=20)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/probe_v6_transport.py'),
                    '--out',str(output),'--pairs','20'],cwd=ROOT,check=True)
    after=interface();probe=json.loads((output/'probe.json').read_text());s=probe['summary']
    passed=network_gate(probe,before,after)
    receipt=dict(schema='v6-progress-network-retest-preflight-v1',verified_at=time.time(),
                 interface_before=before,interface_after=after,gates=gates,passed=passed,
                 probe_complete=probe['complete'],summary=s,
                 scope='No environment steps; direct route is only a precondition, not a latency guarantee during the later pilot.')
    (output/'gate.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
