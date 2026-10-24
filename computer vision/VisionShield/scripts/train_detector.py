"""Optional fine-tuning entrypoint. No custom training has been run for this project."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser=argparse.ArgumentParser(description="Fine-tune on your reviewed YOLO dataset; separate recording sessions for train/val/test")
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--model',default='yolo11s.pt')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if not args.data.is_file():
        parser.error('Supply a real annotated dataset YAML; the project template is not training data.')
    import yaml
    data=yaml.safe_load(args.data.read_text(encoding='utf-8'))
    if not all(data.get(k) for k in ['train','val','names']):
        parser.error('Dataset must declare train, val and names')
    if data['train']==data['val']:
        parser.error('Use separate training and validation recordings; shared splits cannot establish accuracy')
    # Ultralytics resolves relative paths against its dataset root; make this explicit.
    if data.get('path') and not Path(data['path']).is_absolute():
        data['path']=str((args.data.resolve().parent/str(data['path'])).resolve())
    resolved=Path('runs/security/dataset.resolved.yaml')
    resolved.parent.mkdir(parents=True,exist_ok=True)
    resolved.write_text(yaml.safe_dump(data),encoding='utf-8')
    YOLO(args.model).train(data=str(resolved.resolve()),epochs=args.epochs,imgsz=640,device=args.device,
                          seed=42,project='runs/security',name='finetune')

if __name__=='__main__':
    main()
