"""Training-only event-window weights derived from expert phase annotations."""
import numpy as np


def event_window_weights(phases, *, before=15, after=25, boost=2.0):
    phases=np.asarray(phases)
    if phases.ndim!=1 or type(before) is not int or type(after) is not int or min(before,after)<0 or after<1 or not np.isfinite(boost) or boost<1:
        raise ValueError('Invalid event sampling options')
    entry=np.r_[True,phases[1:]!=phases[:-1]] if len(phases) else np.zeros(0,bool)
    events=np.flatnonzero(entry & np.isin(phases,['close_gripper','release']))
    mask=np.zeros(len(phases),bool)
    for index in events:mask[max(0,index-before):min(len(phases),index+after)]=True
    # Overlapping event windows are a union; they never multiply repeatedly.
    return np.where(mask,boost,1.0),events
