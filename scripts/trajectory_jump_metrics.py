"""Offline arm trajectory and same-timestamp raw-plan disagreement diagnostics."""
import numpy as np

ARMS=(0,1,2,3,4,5,7,8,9,10,11,12)


def distribution(values):
    values=np.asarray(values,float)
    return None if not values.size else dict(p50=float(np.quantile(values,.5)),p99=float(np.quantile(values,.99)),max=float(np.max(values)))


def trajectory_jumps(data):
    states=np.asarray(data['state'],float);n=len(states)
    assert states.shape==(n,14) and n>=4 and np.isfinite(states).all()
    indices=np.asarray(data['action_prediction_index'],int);origins=np.asarray(data['prediction_steps'],int)
    chunks=np.asarray(data['predicted_chunks'],float)
    assert indices.shape==(n,) and chunks.ndim==3 and chunks.shape[2]==14 and len(origins)==len(chunks)
    assert indices.min()>=0 and indices.max()<len(chunks) and np.isfinite(chunks).all()
    switches=np.flatnonzero(np.diff(indices)!=0)+1
    result=dict(physical_increment_rad=distribution(np.max(np.abs(np.diff(states[:,ARMS],axis=0)),axis=1)),
        physical_velocity_change_rad_s=distribution(np.max(np.abs(np.diff(states[:,ARMS],n=2,axis=0)),axis=1)/.02),
        switch_count=len(switches),command_switch_increments_rad={},same_time_raw_disagreement_rad=None,
        old_plan_overlap_count=0,old_plan_overlap_unavailable=0,switches=[])
    for name in ('requested_action','blended_action','submitted_action','action'):
        values=np.asarray(data[name],float);assert values.shape==(n,14) and np.isfinite(values).all()
        delta=np.max(np.abs(np.diff(values[:,ARMS],axis=0)),axis=1)
        result['command_switch_increments_rad'][name]=distribution(delta[switches-1])
    disagreements=[]
    for step in switches:
        old,new=int(indices[step-1]),int(indices[step]);a=int(step-origins[old]);b=int(step-origins[new])
        assert 0<=b<chunks.shape[1]
        assert np.allclose(np.asarray(data['requested_action'])[step],chunks[new,b],rtol=1e-6,atol=1e-6)
        valid=0<=a<chunks.shape[1]
        disagreement=float(np.max(np.abs(chunks[new,b,list(ARMS)]-chunks[old,a,list(ARMS)]))) if valid else None
        result['switches'].append(dict(control_step=int(step),old_prediction=old,new_prediction=new,
            old_offset=a,new_offset=b,raw_same_timestamp_max_arm_disagreement_rad=disagreement))
        if valid:disagreements.append(disagreement)
    result['old_plan_overlap_count']=len(disagreements)
    result['old_plan_overlap_unavailable']=len(switches)-len(disagreements)
    result['same_time_raw_disagreement_rad']=distribution(disagreements)
    result['scope']='Physical state increments are pre-step samples. Command switch increments compare consecutive executed timestamps; raw same-time disagreement compares two unblended plans only when the old plan still covers that timestamp. These are distinct from contact forces or causal explanations. Grippers excluded.'
    return result
