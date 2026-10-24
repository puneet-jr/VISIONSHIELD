from pathlib import Path
from hashlib import sha256
import json
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
records=[]
for name in ['yolo11n.pt','yolo11s.pt']:
    path=ROOT/name
    model=YOLO(path)
    args=model.ckpt.get('train_args',{})
    records.append({'file':name,'sha256':sha256(path.read_bytes()).hexdigest(), 'bytes':path.stat().st_size,
                    'source':'https://github.com/ultralytics/assets/releases/download/v8.3.0/'+name,
                    'checkpoint_date':model.ckpt.get('date'), 'checkpoint_data_argument':args.get('data'),
                    'checkpoint_epochs':args.get('epochs'), 'checkpoint_image_size':args.get('imgsz'),
                    'class_count':len(model.names), 'enabled_classes':{str(k):v for k,v in model.names.items()},
                    'custom_training_performed':False})
report={'default':'yolo11s.pt','fast_profile':'yolo11n.pt','training_dataset':'MS COCO 2017 (vendor pretrained)','models':records}
(ROOT/'artifacts/model-card.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
