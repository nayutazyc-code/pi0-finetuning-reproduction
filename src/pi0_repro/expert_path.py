"""Offline expert geometry with C2 corner joins and a smooth cruise clock."""
import math
import numpy as np


class RoundedPath:
    """Use virtual polyline distance; local quintic corners match two derivatives."""

    def __init__(self, points, radius):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError('Expected finite Nx3 path points')
        if not math.isfinite(radius) or radius<=0:
            raise ValueError('Corner radius must be positive')
        delta=np.diff(points,axis=0);length=np.linalg.norm(delta,axis=1)
        if np.any(length<1e-6):raise ValueError('Path segments must be nonzero')
        unit=delta/length[:,None];bounds=np.r_[0.,np.cumsum(length)]
        self.length=float(bounds[-1]);self.bounds=bounds;self.pieces=[]
        trims=np.zeros(len(points))
        for i in range(1,len(points)-1):trims[i]=min(radius,.25*length[i-1],.25*length[i])
        for i in range(len(length)):
            start=bounds[i]+trims[i];end=bounds[i+1]-trims[i+1]
            self.pieces.append((float(start),float(end),points[i]+unit[i]*trims[i],unit[i]))
            if i+1<len(length):
                j=i+1;r=trims[j];p=points[j];a=p-r*unit[i];b=p+r*unit[j]
                controls=np.array([a,a+2*r/5*unit[i],a+4*r/5*unit[i],b-4*r/5*unit[j],b-2*r/5*unit[j],b])
                self.pieces.append((float(bounds[j]-r),float(bounds[j]+r),controls,None))

    def evaluate(self, distance):
        if not math.isfinite(distance):raise ValueError('Distance must be finite')
        s=float(np.clip(distance,0,self.length))
        for lo,hi,data,direction in self.pieces:
            if s<=hi+1e-12:
                if direction is not None:return data+(s-lo)*direction
                u=float(np.clip((s-lo)/(hi-lo),0,1))
                return sum(math.comb(5,k)*(1-u)**(5-k)*u**k*data[k] for k in range(6))
        raise RuntimeError('Path interval missing')


class SmoothCruise:
    """Quintic velocity ramps, bounded positive cruise; position is C3 in time."""

    def __init__(self, distance, max_speed, ramp_seconds=.5):
        if not all(math.isfinite(v) and v>0 for v in (distance,max_speed,ramp_seconds)):
            raise ValueError('Cruise distance, speed and ramp must be finite positive')
        self.distance=float(distance);self.ramp=float(ramp_seconds)
        self.speed=min(float(max_speed),self.distance/self.ramp)
        self.cruise=max(0.,self.distance/self.speed-self.ramp)
        self.duration=2*self.ramp+self.cruise

    @staticmethod
    def _integral(u):return u**6-3*u**5+2.5*u**4

    def evaluate(self, time):
        if not math.isfinite(time):raise ValueError('Time must be finite')
        t=float(np.clip(time,0,self.duration));r=self.ramp;v=self.speed;c=self.cruise
        if t<=r:return v*r*self._integral(t/r)
        if t<=r+c:return .5*v*r+v*(t-r)
        u=(t-r-c)/r
        return min(self.distance,.5*v*r+v*c+v*r*(u-self._integral(u)))
