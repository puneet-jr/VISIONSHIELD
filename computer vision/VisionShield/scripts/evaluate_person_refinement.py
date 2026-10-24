"""Same fixed Mall sample: unmodified full-frame detector vs enlarged person proposals."""
from pathlib import Path
import sys,json,io,zipfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scipy.io import loadmat
from app.engine.detector import ObjectDetector
from app.engine.person_refinement import PersonRefiner


def main():
    detector=ObjectDetector();refiner=PersonRefiner();samples=[]
    with zipfile.ZipFile(ROOT/'datasets/crowd/mall/source.zip') as data:
        annotations=loadmat(io.BytesIO(data.read('mall_dataset/mall_gt.mat')),simplify_cells=True)['frame']
        for index in np.linspace(0,len(annotations)-1,24,dtype=int):
            frame=cv2.imdecode(np.frombuffer(data.read(f'mall_dataset/frames/seq_{index+1:06d}.jpg'),np.uint8),cv2.IMREAD_COLOR)
            expected=len(np.asarray(annotations[index]['loc']).reshape(-1,2))
            base=detector.detect(frame);started=time.perf_counter();refined=refiner.refine(frame,base,key=int(index))
            samples.append({'frame':int(index+1),'annotated_heads':expected,'baseline':sum(d['label']=='person' for d in base),
                'refined':sum(d['label']=='person' for d in refined),'refinement_ms':round((time.perf_counter()-started)*1000,1)})
    metrics=lambda key:{'mae':float(np.mean([abs(s[key]-s['annotated_heads']) for s in samples])),
        'bias':float(np.mean([s[key]-s['annotated_heads'] for s in samples]))}
    blank=refiner.refine(np.zeros((480,640,3),dtype=np.uint8),[])
    report={'dataset':'Mall fixed 24/2000 frames','baseline':metrics('baseline'),'refined':metrics('refined'),
        'extra_inference_ms_median':float(np.median([s['refinement_ms'] for s in samples])),
        'blank_person_detections':sum(d['label']=='person' for d in blank),'samples':samples,
        'trained_on_mall':False,'limitations':['Single-scene diagnostic; head counts and body boxes differ.','Not a crowd safety/identity validation; smaller occluded people may remain undetected.']}
    (ROOT/'artifacts/person-refinement-evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
