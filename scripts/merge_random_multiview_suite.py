"""Bind a transport-aborted prefix and its disjoint continuation without retries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KNOWN_ABORT='AssertionError: Inference underrun; retain and abort'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--continuation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    prefix,continuation,out=(v.resolve() for v in (args.prefix,args.continuation,args.out))
    assert all(v.is_relative_to(ROOT/'outputs') for v in (prefix,continuation,out))
    first,last=read(prefix/'comparison.json'),read(continuation/'comparison.json')
    assert not first['complete'] and first.get('error')==KNOWN_ABORT
    assert last['complete'] and not last.get('error')
    assert first['plan']['source_sha256']==last['plan']['source_sha256']
    assert first['plan']['profile_sha256']==last['plan']['profile_sha256']
    assert last['plan']['continuation_of']['comparison_sha256']==sha(prefix/'comparison.json')
    assert first['plan']['cases'][len(first['results']):]==last['plan']['cases']
    all_rows=copy.deepcopy(first['results']+last['results'])
    assert len(all_rows)==12 and [r['index'] for r in all_rows]==list(range(12))
    assert len({r['seed'] for r in all_rows})==12
    identities=[v[k]['service_identity'] for v in (first,last) for k in ('endpoint_before','endpoint_after')]
    assert all(v==identities[0] for v in identities)
    for row in all_rows:
        row['origin_phase']='initial' if row['index']<len(first['results']) else 'continuation'
        if row.get('infrastructure_error'):
            assert row['infrastructure_error']==KNOWN_ABORT
            execution=row['diagnostics']['episodes'][0]['result']['execution']
            assert execution['inference_underruns']>0 and not row['diagnostics']['episodes'][0]['result'].get('error')
            row['accepted_transport_warning']=True
            row['transport_warning']={'underruns':execution['inference_underruns'],
                'wait_seconds':execution['underrun_wait_seconds'],
                'scope':'Original strict wrapper aborted after the policy episode had completed; its outcome and wait remain included.'}
    plan=copy.deepcopy(first['plan'])
    plan.update(schema='v6-random-planar-multiview-combined-v1',script_sha256=sha(Path(__file__)),
        phase_count=2,parts=[{'path':str(v),'comparison_sha256':sha(v/'comparison.json'),'plan_sha256':sha(v/'plan.json'),
                             'runner_sha256':sha(v/'runner_source.py')} for v in (prefix,continuation)],
        protocol_note='Initial strict timing gate aborted after a completed task with an underrun. Remaining seeds ran once, disjointly, retaining transport warnings. This is a two-phase descriptive capability suite, not an uninterrupted timing comparison.',
        manifest_builder_scope='runner_source.py here is the manifest builder. Each phase retains its actual policy-run harness.',
        retain_transport_warnings=True)
    out.mkdir(parents=True,exist_ok=False)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'plan.json',plan)
    write(out/'comparison.json',{'plan':plan,'complete':True,'results':all_rows,
        'endpoint_before':first['endpoint_before'],'endpoint_after':last['endpoint_after'],
        'phase_count':2,'all_original_phase_ledgers_preserved':True,'no_seed_repeated':True})
    print(json.dumps({'out':str(out),'attempts':12,'successes':sum(bool(r['success']) for r in all_rows),
                      'transport_warning_attempts':[r['index'] for r in all_rows if r.get('transport_warning')]},indent=2))


if __name__=='__main__':main()
