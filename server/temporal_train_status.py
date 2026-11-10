"""Read-only aggregate progress; never export normalization or model tensors."""
import json
from pathlib import Path
import re
import time


root=Path(__file__).resolve().parents[1]
out=root/'outputs/v8_temporal_20261026'
path=out/'status.json'
if not path.exists():
    print(json.dumps({'status':'not_started'}))
else:
    status=json.loads(path.read_text())
    result={k:status.get(k) for k in ('status','active_arm','training_order','error','restore_v6_attempt_pid','restore_v6_verified')}
    result['elapsed_seconds']=time.time()-status['started_at']
    result['arms']=[]
    for arm in ('control','temporal'):
        row={'arm':arm}
        probe=out/(arm+'_probe.json')
        if probe.exists():
            p=json.loads(probe.read_text());row['probe']={k:p.get(k) for k in ('status','completed_optimizer_updates','actual_action_shape','steady_update_seconds','gpu_peak_used_mib','gpu_total_mib','error')}
        log=out/(arm+'_training.log')
        if log.exists():
            text=log.read_text(errors='replace');steps=[int(s) for s in re.findall(r'Step (\d+):',text)]
            row['last_logged_step']=max(steps,default=None)
            row['completed_updates_lower_bound']=max([int(v) for v in re.findall(r'(\d+)/3000',text)]+[s+1 for s in steps],default=0)
            row['nonfinite_metric_lines']=sum(bool(re.search(r'\b(?:nan|[+-]?inf)\b',line,re.I)) for line in text.splitlines() if re.search(r'Step \d+:',line))
            row['last_metric_line']=next((s for s in reversed(text.splitlines()) if re.search(r'Step \d+:',s)),None)
        final=root/'checkpoints/pi0_banana_lora'/f'banana_v8_{arm}32_3000/2999'
        row['final_checkpoint_committed']=(final/'_CHECKPOINT_METADATA').is_file() and (final/'params/_METADATA').is_file()
        result['arms'].append(row)
    reader=out/'reader_audit.json'
    if reader.exists():
        a=json.loads(reader.read_text());result['reader_audit']={k:v for k,v in a.items() if k!='samples'}
    print(json.dumps(result))
