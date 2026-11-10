"""Frozen 24-attempt two-profile acceptance; no training or automatic promotion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from compare_v6_rtc import LAYOUTS
from evaluate_release_replication import ALL_SOURCE_NAMES
from profile_v6_runtime import inspect_run
from analyze_release_replication import summarize
from analyze_rtc_pair import equal

BLOCKS = ('v6', 'candidate_damping2', 'candidate_damping2', 'v6')
NORM = '6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
MODELS = {
    'v6': dict(checkpoint_model='v6', damping_scale=1., suffix='banana_v6_long100_accum8000/7999'),
    'candidate_damping2': dict(checkpoint_model='candidate', damping_scale=2., suffix='banana_v6_ik_candidate1000/999'),
}
GATE = dict(minimum_successes=10, success_shortfall=0, typical_jerk_ratio=.8,
            worst_peak_ratio=1., increment_ratio=1.05, pause_add_seconds=.1,
            duration_ratio=1.1, wall_ratio=1.1, apply_p99_ratio=1.1, apply_max_ratio=1.1,
            wait_add_seconds=.1, minimum_transport_coverage=10)
SCRIPT_DEPS = ('scripts/evaluate_final_candidate.py', 'scripts/profile_v6_runtime.py',
               'scripts/analyze_release_replication.py', 'scripts/analyze_rtc_pair.py',
               'scripts/summarize_banana_execution.py', 'scripts/trajectory_jump_metrics.py',
               'scripts/evaluate_release_replication.py', 'scripts/evaluate_random_multiview.py',
               'scripts/compare_v6_rtc.py')

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, d):
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    q.replace(p)

def make_cases():
    cases, seen = [], Counter()
    for block, model in enumerate(BLOCKS):
        seen[model] += 1
        order = list(LAYOUTS) if block < 2 else list(reversed(LAYOUTS))
        order = order[block:] + order[:block]
        for layout in order:
            cases.append(dict(index=len(cases), block=block, model=model, layout=layout, repeat=seen[model]))
    return cases

def gate_checks(candidate, reference, gate=GATE):
    checks = []
    def valid(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    def compare(name, key, factor=1., add=0., positive=True):
        a, b = reference.get(key), candidate.get(key)
        ok = valid(a) and valid(b) and (a > 0 if positive else a >= 0) and (b > 0 if positive else b >= 0)
        checks.append(dict(name=name, reference=a, candidate=b, passed=bool(ok and b <= a*factor+add)))
    for label, key, ratio in [
        ('typical_jerk', 'median_active_jerk_p99', gate['typical_jerk_ratio']),
        ('covered_typical_jerk', 'covered_median_active_jerk_p99', gate['typical_jerk_ratio']),
        ('worst_active_peak', 'worst_active_jerk_peak', gate['worst_peak_ratio']),
        ('worst_allframe_peak', 'worst_all_frame_jerk_peak', gate['worst_peak_ratio']),
        ('worst_increment', 'worst_joint_increment_rad', gate['increment_ratio']),
        ('covered_sim_duration', 'covered_median_simulated_seconds', gate['duration_ratio']),
        ('covered_wall_duration', 'covered_median_wall_seconds', gate['wall_ratio']),
        ('apply_p99', 'median_trial_apply_p99_ms', gate['apply_p99_ratio']),
        ('apply_max', 'worst_apply_interval_ms', gate['apply_max_ratio'])]:
        compare(label, key, ratio)
    compare('longest_pause', 'worst_low_motion_pause_seconds', add=gate['pause_add_seconds'], positive=False)
    compare('median_total_pause', 'median_low_motion_pause_total_seconds', add=gate['pause_add_seconds'], positive=False)
    compare('underruns', 'total_running_inference_underruns', positive=False)
    compare('running_wait', 'total_running_wait_seconds', add=gate['wait_add_seconds'], positive=False)
    for key in ('attempts', 'scored_active_motion'):
        checks.append(dict(name=key, passed=candidate.get(key)==reference.get(key)==12))
    for key, minimum in [('successes',gate['minimum_successes']), ('transport_covered',gate['minimum_transport_coverage'])]:
        a,b=reference.get(key),candidate.get(key)
        checks.append(dict(name=key, reference=a,candidate=b, minimum=minimum,
                           passed=valid(a) and valid(b) and b>=max(minimum,a)))
    return checks

def prepare(out):
    receipt = ROOT/'outputs/final_candidate_preflight_20261030/remote_receipt.json'
    remote = read(receipt)
    assert all(remote['checkpoints'][name]['norm_sha256']==NORM for name in ('v6','candidate'))
    training = read(ROOT/'outputs/ik_continuous_20261029/training_plan_train_only.json')
    dataset = ROOT/training['arms'][0]['dataset']
    mapping_path = dataset/'local/pi0_banana_validation/source_episodes.json'
    mapping = read(mapping_path)
    assert mapping['source_split']=='validation' and set(LAYOUTS)<={r['source_episode'] for r in mapping['episodes'] if r['split']=='validation'}
    raw = Path(read(dataset/'dataset_manifest.json')['source_path'])
    layouts = {n:read(raw/n/'layout.json') for n in LAYOUTS}
    paths = [*(f'src/pi0_repro/{n}' for n in ALL_SOURCE_NAMES), *SCRIPT_DEPS,
             'configs/banana_v6_bridge20_pi0.toml']
    plan = dict(schema='final-candidate-acceptance-v1', attempts=24, cases=make_cases(), models=MODELS,
                gate=GATE, norm_stats_sha256=NORM, noise_paired=False, layouts=layouts,
                source_sha256={n:sha(ROOT/n) for n in sorted(set(paths))},
                validation_mapping=str(mapping_path.relative_to(ROOT)),validation_mapping_sha256=sha(mapping_path),
                remote_preflight=str(receipt.relative_to(ROOT)), remote_preflight_sha256=sha(receipt),
                checkpoint_manifests=remote['checkpoints'],server_sources=remote['server_sources'],
                network_scope='Current unchanged route, including TUN if present; separate from historical no-TUN cohorts. No VPN mutation. Abort on interface change.',
                warmup=dict(requests_per_block=6,scored_last=4,median_rtt_ms_max=800,max_rtt_ms_max=1500),
                runtime='H100, denoise10, RTC20/client bridge20, 3Hz arm filter, headless top25fps, original limits, max40 simulation seconds; no training.',
                decision='All new gates plus audit and primary visual review required. No automatic promotion, favorable reruns, or pooling old cohorts.')
    if out.exists():
        assert read(out/'plan.json')==plan,'Existing frozen plan differs'
    else:
        out.mkdir(parents=True)
        write(out/'plan.json',plan)
        for n in plan['source_sha256']:
            p=out/'frozen_source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
        write(out/'remote_preflight.json',remote)
        for name,row in layouts.items():
            for scale in (1.,2.):
                scene=dict(row['scene'],wrist_damping_scale=scale)
                p=out/f'{name}_damping{int(scale)}.toml'
                p.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in scene.items() if v is not None)+'\n')
    return plan

def connection_config(plan,model,port):
    """Metadata checks must remain usable even if a scene fails validation."""
    cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml')
    cfg.connection.port=port;cfg.connection.connect_timeout_seconds=5
    cfg.model.expected_checkpoint_suffix=plan['models'][model]['suffix']
    return cfg

def configured(plan,out,model,layout,port):
    cfg=connection_config(plan,model,port)
    s=cfg.simulation;s.episodes=1;s.scene_config=str(out/f'{layout}_damping{int(plan["models"][model]["damping_scale"])}.toml')
    s.seed=plan['layouts'][layout]['reset_seed'];s.max_steps=2000;s.display=False;s.save_video=True;s.video_fps=25
    from dataclasses import asdict
    from pi0_repro.scene import SceneConfig,load_scene_config
    expected=SceneConfig(**dict(plan['layouts'][layout]['scene'],wrist_damping_scale=plan['models'][model]['damping_scale']))
    # JSON/TOML sequences become lists and validated SceneConfig uses tuples.
    # Compare their exact JSON values; do not loosen numeric/field equality.
    canonical=lambda scene: json.loads(json.dumps(asdict(scene),allow_nan=False))
    assert canonical(load_scene_config(Path(s.scene_config)))==canonical(expected),'Generated scene changed'
    return cfg

def unchanged(plan):
    assert all(sha(ROOT/n)==v for n,v in plan['source_sha256'].items()),'Source/config/analyzer changed'
    assert sha(ROOT/plan['remote_preflight'])==plan['remote_preflight_sha256']
    assert sha(ROOT/plan['validation_mapping'])==plan['validation_mapping_sha256']

def run_suite(out,plan):
    import numpy as np
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    assert not (out/'comparison.json').exists(),'Existing attempts retained; no rerun'
    unchanged(plan)
    ledger=dict(plan=plan,complete=False,results=[],selected_models=[],routes=[],restoration=None)
    write(out/'comparison.json',ledger)
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    remote='/root/pi0-repro'
    def route():
        raw=subprocess.check_output(['route','-n','get',target],text=True,timeout=15)
        match=re.search(r'interface:\s*(\S+)',raw);assert match,'Route interface unavailable'
        interface=match.group(1)
        assert not ledger.get('route_interface') or ledger['route_interface']==interface,'Route changed; preserve partial cohort'
        ledger['route_interface']=interface
        return interface
    def select(model):
        cmd=[remote+'/openpi/.venv/bin/python','-u',remote+'/outputs/ik_continuous_20261029/cloud_policy_select.py','--root',remote,'--model',MODELS[model]['checkpoint_model']]
        r=subprocess.run(ssh+[target,shlex.join(cmd)],capture_output=True,text=True,timeout=75)
        if r.returncode:raise RuntimeError(r.stderr[-1800:].replace(target,'[saved target]'))
        return json.loads(r.stdout)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    def endpoint(model):
        deadline=time.monotonic()+360
        while True:
            try:
                with RemotePi0(connection_config(plan,model,port)) as client:
                    m=client.metadata
                    assert m['denoise_steps']==10 and m['rtc']['adapter_sha256']==plan['server_sources']['server/rtc_policy.py']
                    return m
            except (OSError,TimeoutError):
                if time.monotonic()>=deadline:raise
                time.sleep(1)
            except RuntimeError as e:
                if not any(s in str(e) for s in ('ConnectionRefusedError','connection refused')) or time.monotonic()>=deadline:raise
                time.sleep(1)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')])
    os.environ.setdefault('MUJOCO_GL','glfw')
    tunnel=awake=None;changed=False
    log=(out/'tunnel.log').open('x')
    try:
        # Connecting to a missing sidecar is not permission to replace an unknown process.
        probe="import socket,json;s=socket.socket();s.settimeout(2);print(json.dumps({'proxy_listening':s.connect_ex(('127.0.0.1',8001))==0}));s.close()"
        r=subprocess.run(ssh+[target,shlex.join([remote+'/openpi/.venv/bin/python','-c',probe])],text=True,capture_output=True,timeout=25)
        assert r.returncode==0 and json.loads(r.stdout)['proxy_listening'],'Proxy8001 missing; primary agent must restore it'
        route()
        tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{port}:127.0.0.1:8001',target],stdout=log,stderr=subprocess.STDOUT)
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
        for block,model in enumerate(BLOCKS):
            unchanged(plan);ledger['routes'].append(dict(block=block,interface=route()))
            changed=True;dispatch=select(model);metadata=endpoint(model)
            ledger['selected_models'].append(dict(block=block,model=model,dispatch=dispatch,metadata=metadata));write(out/'comparison.json',ledger)
            cfg=configured(plan,out,model,LAYOUTS[0],port);warm=out/f'warmup_{block:02d}';warm.mkdir();env=_make_env(cfg,warm);requests=[]
            try:
                obs,_=env.reset(seed=cfg.simulation.seed)
                observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
                prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
                with RemotePi0(cfg) as client:
                    for _ in range(6):
                        actions,ms,response=client.infer({**observation,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)})
                        assert actions.shape==(100,14) and np.isfinite(actions).all()
                        requests.append(dict(round_trip_ms=ms,server_timing=response.get('server_timing')))
            finally:env.close();write(warm/'requests.json',dict(no_physical_steps=True,requests=requests))
            assert max(v['round_trip_ms'] for v in requests[2:])<1500 and np.median([v['round_trip_ms'] for v in requests[2:]])<800,'Readiness gate failed; no favorable rerun'
            for case in [c for c in plan['cases'] if c['block']==block]:
                unchanged(plan);row=dict(case,route=route());ledger['results'].append(row);write(out/'comparison.json',ledger)
                print('START',case,flush=True)
                try:
                    p=run(configured(plan,out,model,case['layout'],port));row['run_directory']=str(p)
                    row['diagnostics']=inspect_run(p);d=row['diagnostics']
                    assert d['prefix_reconstruction']['passed'] and d['summary']['raw_prediction_max_error']==0 and d['summary']['max_action_timestamp_error_steps']==0
                    assert not d['summary']['result']['error'] and not d['summary']['result']['aborted_by_window_close']
                except BaseException as e:row['operational_error']=f'{type(e).__name__}: {e}'.replace(target,'[saved target]');raise
                finally:write(out/'comparison.json',ledger)
                print('END',case['index'],d['summary']['result']['success'],d['action_apply_ms'],flush=True)
        ledger['complete']=True
    except BaseException as e:ledger['error']=f'{type(e).__name__}: {e}'.replace(target,'[saved target]');raise
    finally:
        try:
            if changed:
                try:ledger['restoration']=dict(dispatch=select('v6'),metadata=endpoint('v6'),verified=True)
                except BaseException as e:ledger['restoration']=dict(verified=False,error=str(e).replace(target,'[saved target]'))
        finally:
            write(out/'comparison.json',ledger)
            if tunnel is not None:
                tunnel.terminate()
                try:tunnel.wait(timeout=5)
                except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
            if awake is not None:awake.terminate()
            log.close()
    print('COMPLETE',out,flush=True)

def xml_pair_equal(base,candidate):
    left,right=ET.fromstring(base),ET.fromstring(candidate);changes=[]
    aa,bb=list(left.iter()),list(right.iter())
    if len(aa)!=len(bb):return False
    expected={f'vx300s_{side}/{kind}' for side in ('left','right') for kind in ('forearm_roll','wrist_angle','wrist_rotate')}
    for a,b in zip(aa,bb):
        if a.attrib!=b.attrib:
            if a.tag!=b.tag or a.tag!='joint' or a.get('name') not in expected:return False
            if {k for k in set(a.attrib)|set(b.attrib) if a.get(k)!=b.get(k)}!={'damping'}:return False
            if float(b.get('damping'))!=2*float(a.get('damping')):return False
            changes.append(a.get('name'));b.set('damping',a.get('damping'))
    return set(changes)==expected and len(changes)==6 and ET.tostring(left)==ET.tostring(right)

def analyze(out,plan):
    unchanged(plan)
    assert not (out/'aggregate.json').exists() and not (out/'audit.json').exists(),'Retain prior analysis'
    ledger=read(out/'comparison.json');checks=[]
    def check(name,passed):checks.append(dict(name=name,passed=bool(passed)))
    check('complete24',ledger.get('complete') and len(ledger['results'])==24)
    check('frozen_plan',ledger['plan']==plan)
    check('restored_v6',(ledger.get('restoration') or {}).get('verified'))
    for n,h in plan['source_sha256'].items():
        p=out/'frozen_source'/n
        check('frozen_source:'+n,p.is_file() and sha(p)==h)
    expected=Counter((c['model'],c['layout'],c['repeat']) for c in plan['cases'])
    check('case_set',Counter((c['model'],c['layout'],c['repeat']) for c in ledger['results'])==expected)
    check('unique_runpaths',len({r.get('run_directory') for r in ledger['results']})==24)
    check('block_selections',len(ledger['selected_models'])==4 and [r['model'] for r in ledger['selected_models']]==list(BLOCKS))
    check('route_stable',bool(ledger.get('route_interface')) and all(r['interface']==ledger['route_interface'] for r in ledger['routes']))
    groups=defaultdict(list);scenes=defaultdict(list);xmls=defaultdict(dict)
    for block in range(4):
        warm=out/f'warmup_{block:02d}/requests.json'
        if not warm.exists():check(f'warmup{block}',False);continue
        requests=read(warm)['requests'];values=[r['round_trip_ms'] for r in requests[2:]]
        import numpy as np
        check(f'warmup{block}',len(requests)==6 and max(values)<1500 and np.median(values)<800)
    for row in ledger['results']:
        i=row['index'];case=plan['cases'][i];check(f'{i}:case',all(row[k]==v for k,v in case.items()))
        check(f'{i}:route',row.get('route')==ledger.get('route_interface'))
        if 'run_directory' not in row:check(f'{i}:run',False);continue
        p=Path(row['run_directory']);actual=inspect_run(p);meta=read(p/'server.json');cfg=read(p/'run.json')['config'];s=cfg['simulation']
        check(f'{i}:recomputed',equal(actual,row['diagnostics']))
        check(f'{i}:prefix',actual['prefix_reconstruction']['passed'])
        check(f'{i}:timestamp',actual['summary']['raw_prediction_max_error']==actual['summary']['max_action_timestamp_error_steps']==0)
        result=actual['summary']['result'];check(f'{i}:operational',not row.get('operational_error') and result['error'] is None and not result['aborted_by_window_close'])
        check(f'{i}:metadata',meta['checkpoint'].endswith('/'+MODELS[row['model']]['suffix']) and meta['norm_stats_sha256']==NORM and meta['model_family']=='pi0' and meta['action_dim']==14 and meta['action_horizon']==100 and meta['denoise_steps']==10 and meta['rtc']['adapter_sha256']==plan['server_sources']['server/rtc_policy.py'])
        expected_cfg=configured(plan,out,row['model'],row['layout'],cfg['connection']['port'])
        from dataclasses import asdict
        check(f'{i}:config',cfg==asdict(expected_cfg))
        for field in ('source_code_sha256','optional_source_code_sha256'):
            for n,h in read(p/'run.json')[field].items():check(f'{i}:source:{n}',plan['source_sha256'].get('src/pi0_repro/'+n)==h)
        scene=read(p/'episode_000_scene.json');check(f'{i}:scale',scene['scene']['wrist_damping_scale']==MODELS[row['model']]['damping_scale'])
        scene['scene'].pop('wrist_damping_scale');scene['reset']['scene'].pop('wrist_damping_scale');scenes[row['layout']].append(scene)
        xmls[row['layout']].setdefault(row['model'],[]).append((p/'scene.xml').read_text())
        video=p/'episode_000.mp4';check(f'{i}:video',video.is_file() and video.stat().st_size>0)
        e=result['execution'];trial=dict(index=i,model=row['model'],layout=row['layout'],repeat=row['repeat'],run_directory=str(p),success=result['success'],simulated_seconds=e['simulated_seconds'],control_wall_seconds=e['control_wall_seconds'],inference_underruns=e['inference_underruns'],inference_wait_seconds=e['underrun_wait_seconds'],action_apply_ms=actual['action_apply_ms'],quality=actual['motion_quality'],video=str(video),video_sha256=sha(video))
        groups[row['model']].append(trial)
    for layout in LAYOUTS:
        rs=scenes[layout];check('same_reset:'+layout,len(rs)==4 and all(equal(r,rs[0]) for r in rs))
        xs=xmls[layout];a=xs.get('v6',[]);b=xs.get('candidate_damping2',[])
        check('six_damping_only:'+layout,len(a)==len(b)==2 and a[0]==a[1] and b[0]==b[1] and xml_pair_equal(a[0],b[0]))
    models={k:summarize(groups[k]) for k in MODELS}
    gates=gate_checks(models['candidate_damping2'],models['v6'],plan['gate'])
    audit=dict(passed=all(c['passed'] for c in checks),checks=checks,ledger_sha256=sha(out/'comparison.json'),plan_sha256=sha(out/'plan.json'))
    passed=audit['passed'] and all(c['passed'] for c in gates)
    aggregate=dict(complete=ledger.get('complete',False),audit_passed=audit['passed'],models=models,gates=gates,passed=passed,decision='eligible_for_primary_visual_review' if passed else 'retain_v6_reference',scope=plan['network_scope'],trials=[r for k in MODELS for r in groups[k]])
    write(out/'audit.json',audit);write(out/'aggregate.json',aggregate)
    print(json.dumps(dict(passed=passed,models=models,failed_checks=[c for c in checks+gates if not c['passed']]),indent=2))
    return aggregate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--prepare-only',action='store_true');group.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/final_candidate_acceptance_20261030')
    args=parser.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs') and out!=ROOT/'outputs'
    plan=prepare(out)
    if args.prepare_only:print('PREPARED',out);return
    if args.analyze_only:analyze(out,plan)
    else:run_suite(out,plan)

if __name__=='__main__':main()
