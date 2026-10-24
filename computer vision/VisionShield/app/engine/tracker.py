"""Stateful Multi-Object Tracking Engine
Provides persistent object IDs across frames using class-aware centroid distance matching.
Maintains trajectory histories, smoothed speed, and dwell counters.
"""
import numpy as np
import cv2
import time
from typing import List, Dict, Any, Tuple, Optional

class TrackedObject:
    def __init__(self, track_id: int, detection: Dict[str, Any], timestamp: float):
        self.track_id = track_id
        self.label = detection["label"]
        self.class_id = detection["class_id"]
        self.box = list(detection["box"])
        self.centroid = detection["centroid"]
        self.confidence = detection["confidence"]
        self.aspect_ratio = detection["aspect_ratio"]
        self.previous_aspect_ratio = self.aspect_ratio
        
        # Temporal & Trajectory History
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.centroid_history: List[Tuple[float, float, float]] = [(self.centroid[0], self.centroid[1], timestamp)]
        self.disappeared_count = 0
        
        # Kinematics
        self.motion = np.zeros(2, dtype=np.float32)
        self.appearance = None
        self.reacquired = False
        self.velocity_px_s = 0.0
        self.is_stationary = False
        
        # Association / Threat States
        self.owner_id: Optional[int] = None
        self.abandoned_since: Optional[float] = None
        self.threat_score: float = 0.0
        self.threat_level: str = "LOW"
        self.active_alert: Optional[str] = None

    def update(self, detection: Dict[str, Any], timestamp: float):
        old_cx, old_cy = self.centroid
        new_cx, new_cy = detection["centroid"]
        
        dt = max(1e-4, timestamp - self.last_seen)
        dx = new_cx - old_cx
        dy = new_cy - old_cy
        dist = np.hypot(dx, dy)
        
        if dt > 1.0:
            self.centroid_history.clear()
            self.first_seen = timestamp
            self.motion[:] = 0
        else:
            self.motion = .5 * self.motion + .5 * np.array([dx/dt, dy/dt], dtype=np.float32)

        # Exponential moving average for velocity
        current_vel = dist / dt
        self.velocity_px_s = 0.7 * self.velocity_px_s + 0.3 * current_vel
        self.is_stationary = (self.velocity_px_s < 15.0)

        self.box = list(detection["box"])
        self.centroid = (new_cx, new_cy)
        self.confidence = detection["confidence"]
        self.previous_aspect_ratio = self.aspect_ratio
        self.aspect_ratio = detection["aspect_ratio"]
        self.last_seen = timestamp
        self.disappeared_count = 0
        
        self.centroid_history.append((new_cx, new_cy, timestamp))
        if len(self.centroid_history) > 60:
            self.centroid_history.pop(0)

    @property
    def dwell_time(self) -> float:
        return self.last_seen - self.first_seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "box": [round(b, 1) for b in self.box],
            "centroid": (round(self.centroid[0], 1), round(self.centroid[1], 1)),
            "velocity": round(self.velocity_px_s, 1),
            "motion": [round(float(v), 1) for v in self.motion],
            "confidence": self.confidence,
            "reacquired": self.reacquired,
            "identity_scope": "session track, not verified identity",
            "dwell_time": round(self.dwell_time, 1),
            "threat_score": round(self.threat_score, 1),
            "threat_level": self.threat_level,
            "owner_id": self.owner_id,
            "active_alert": self.active_alert
        }


def appearance_descriptor(frame, box, person=False):
    """Small torso-colour signature, kept in memory only; no face recognition."""
    if frame is None:
        return None
    h,w=frame.shape[:2]
    x1,y1,x2,y2=map(float,box)
    if person:
        bw,bh=x2-x1,y2-y1
        x1,x2=x1+.15*bw,x2-.15*bw
        y1,y2=y1+.2*bh,y1+.75*bh
    x1,x2=max(0,int(x1)),min(w,int(x2))
    y1,y2=max(0,int(y1)),min(h,int(y2))
    if x2-x1<3 or y2-y1<3:
        return None
    crop=cv2.resize(frame[y1:y2,x1:x2],(24,48))
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    colour=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).flatten()
    brightness=cv2.calcHist([hsv],[2],None,[16],[0,256]).flatten()
    colour/=max(np.linalg.norm(colour),1e-6)
    brightness/=max(np.linalg.norm(brightness),1e-6)
    descriptor=np.concatenate([colour*.8,brightness*.6])
    return descriptor/max(np.linalg.norm(descriptor),1e-6)


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,(a[2]-a[0])*(a[3]-a[1]))+max(0,(b[2]-b[0])*(b[3]-b[1]))-intersection
    return intersection/max(union,1e-6)


class ObjectTracker:
    def __init__(self,max_disappeared=25,max_distance=120.,retention_seconds=30.,max_tracks=128):
        self.next_id=1
        self.tracks={}
        self.max_disappeared=max_disappeared  # retained API compatibility; expiry is time-based
        self.max_distance=max_distance
        self.retention_seconds=retention_seconds
        self.max_tracks=max_tracks

    def update(self,detections,timestamp=None,frame=None):
        timestamp=time.time() if timestamp is None else timestamp
        for tid,track in list(self.tracks.items()):
            ttl=self.retention_seconds if track.label=='person' else 3.
            if timestamp-track.last_seen>ttl:
                del self.tracks[tid]
        descriptors=[appearance_descriptor(frame,d['box'],d['label']=='person') for d in detections]
        edges=[]
        for j,detection in enumerate(detections):
            candidates=[]
            for tid,track in self.tracks.items():
                if track.label!=detection['label']:
                    continue
                gap=max(0,timestamp-track.last_seen)
                delta=track.motion*min(gap,.75)
                predicted=[track.box[0]+delta[0],track.box[1]+delta[1],track.box[2]+delta[0],track.box[3]+delta[1]]
                pc=np.array(track.centroid)+delta
                distance=float(np.linalg.norm(pc-np.array(detection['centroid'])))
                scale=max(self.max_distance,.8*np.hypot(track.box[2]-track.box[0],track.box[3]-track.box[1]))
                overlap=box_iou(predicted,detection['box'])
                signature=descriptors[j]
                app=1-float(np.clip(np.dot(track.appearance,signature),0,1)) if track.appearance is not None and signature is not None else None
                if app is not None and app>.5:
                    continue
                lost=track.disappeared_count>0 or gap>1.
                # Continuity through short gaps is spatially constrained.
                if gap<=2 and distance<=scale and (not lost or distance<scale*.65):
                    cost=.55*(distance/scale)+.2*(1-overlap)+1.0*(app if app is not None else .05)
                    candidates.append((cost,tid,j,False,app))
                elif track.label=='person' and (lost or distance<=scale*2.5) and app is not None and app<.14:
                    old_height=max(1,track.box[3]-track.box[1])
                    ratio=(detection['box'][3]-detection['box'][1])/old_height
                    if .5<=ratio<=2:
                        candidates.append((.3+app,tid,j,True,app))
            reentries=sorted((e for e in candidates if e[3]),key=lambda e:e[4])
            # Similar clothing on multiple lost people is ambiguous: prefer a new ID.
            if len(reentries)>1 and reentries[1][4]-reentries[0][4]<.04:
                candidates=[e for e in candidates if not e[3]]
            edges.extend(candidates)
        assigned_tracks=set();assigned_detections=set()
        for cost,tid,j,reentry,app in sorted(edges,key=lambda edge:edge[0]):
            if tid in assigned_tracks or j in assigned_detections:
                continue
            track=self.tracks[tid]
            track.reacquired=track.disappeared_count>0 or timestamp-track.last_seen>1
            track.update(detections[j],timestamp)
            if descriptors[j] is not None:
                track.appearance=descriptors[j] if track.appearance is None else .8*track.appearance+.2*descriptors[j]
                track.appearance/=max(np.linalg.norm(track.appearance),1e-6)
            assigned_tracks.add(tid);assigned_detections.add(j)
        for tid,track in self.tracks.items():
            if tid not in assigned_tracks:
                track.disappeared_count+=1
        for j,detection in enumerate(detections):
            if j not in assigned_detections:
                self._register(detection,timestamp,descriptors[j])
        if len(self.tracks)>self.max_tracks:
            oldest=sorted(self.tracks,key=lambda tid:self.tracks[tid].last_seen)
            for tid in oldest[:len(self.tracks)-self.max_tracks]:
                del self.tracks[tid]
        return list(self.tracks.values())

    def _register(self,detection,timestamp,appearance=None):
        track=TrackedObject(self.next_id,detection,timestamp)
        track.appearance=appearance
        self.tracks[self.next_id]=track
        self.next_id+=1
