"""Learned person proposals from overlapping enlarged views, deduplicated globally.
This refines visible body detections, not head annotations or unique attendance.
"""
from types import SimpleNamespace
import logging
from app.config import ROOT,config
from app.engine.accelerated_detector import AcceleratedYOLO
import cv2
import numpy as np


def merge_people(detections,iou=.5):
    people=[d for d in detections if d['label']=='person']
    other=[d for d in detections if d['label']!='person']
    if not people:return other
    boxes=[[d['box'][0],d['box'][1],d['box'][2]-d['box'][0],d['box'][3]-d['box'][1]] for d in people]
    indices=cv2.dnn.NMSBoxes(boxes,[float(d['confidence']) for d in people],0.,iou)
    return other+[people[int(index)] for index in np.asarray(indices,dtype=int).reshape(-1)]


class PersonRefiner:
    def __init__(self):
        self.model=None;self.error=None

    def refine(self,frame,base):
        if self.model is None:
            try:
                options=SimpleNamespace(CPU_THREADS=min(2,config.CPU_THREADS),INPUT_SIZE=config.PERSON_TILE_INPUT_SIZE,CONFIDENCE_THRESHOLD=.25,IOU_THRESHOLD=.5,MAX_DETECTIONS=500)
                self.model=AcceleratedYOLO(ROOT/'yolo11n.pt',options)
            except Exception as exc:
                self.error=str(exc)
                logging.getLogger(__name__).warning('Person refinement unavailable: %s',exc)
                return base
        h,w=frame.shape[:2];proposals=list(base)
        for row in range(2):
            for col in range(2):
                x0=max(0,round((col*.5-.12)*w));x1=min(w,round(((col+1)*.5+.12)*w))
                y0=max(0,round((row*.5-.12)*h));y1=min(h,round(((row+1)*.5+.12)*h))
                for detection in self.model.detect(frame[y0:y1,x0:x1],[0]):
                    x,y,x2,y2=detection['box']
                    # Internal top cuts can produce a second lower-body box for
                    # someone already detected above. Keep the complete upper body.
                    if y0 and y<2:continue
                    anchor_x=(x+x2)/2+x0;anchor_y=y+.12*(y2-y)+y0
                    if not (col*w/2<=anchor_x<(col+1)*w/2 and row*h/2<=anchor_y<(row+1)*h/2):continue
                    proposal={**detection,'box':[x+x0,y+y0,x2+x0,y2+y0],
                        'centroid':((x+x2)/2+x0,(y+y2)/2+y0),'detection_source':'enlarged_person_tile'}
                    proposals.append(proposal)
        self.error=None
        return merge_people(proposals)


person_refiner=PersonRefiner()
