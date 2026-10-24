"""Conservative one-time phone orientation check using detector evidence."""
import cv2


def orient_phone(frame,base,detect):
    def people(items):return [item for item in items if item['label']=='person']
    original=people(base);best=(0,frame,base)
    if len(original)>5:return best
    for angle,code in [(90,cv2.ROTATE_90_CLOCKWISE),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        rotated=cv2.rotate(frame,code);detections=detect(rotated)
        observed=people(detections)
        best_people=people(best[2])
        if len(observed)>=len(original)+3 and len(observed)>=1.8*max(1,len(original)):
            if len(observed)>len(best_people) or (len(observed)==len(best_people) and sum(d['confidence'] for d in observed)>sum(d['confidence'] for d in best_people)):
                best=(angle,rotated,detections)
    return best
