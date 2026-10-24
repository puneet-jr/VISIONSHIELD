"""Offline preview/inference isolation check; opens no network ports or cameras."""
from pathlib import Path
import sys,time,json,threading
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.engine.pipeline import SurveillancePipeline
from app.ingestion.video_stream import video_manager
from app.config import config
config.ENABLE_AI_DOSSIER=False
pipeline=SurveillancePipeline()
frame=np.full((480,640,3),55,dtype=np.uint8)
pipeline.detector.detect(frame) # exclude cold model loading from sustained measurements
stopped=threading.Event()
processed=[]
def inference():
    while not stopped.is_set():
        before=pipeline.last_frame_id
        with pipeline.lock:pipeline.run_cycle()
        if pipeline.last_frame_id!=before:processed.append(time.perf_counter())
        stopped.wait(.005)
video_manager.start('phone_upload')
thread=threading.Thread(target=inference,daemon=True);thread.start()
first=video_manager.jpeg_id;started=time.perf_counter()
try:
    while time.perf_counter()-started<2.5:
        video_manager.push_phone_frame(frame)
        time.sleep(1/30)
    elapsed=time.perf_counter()-started
    result={'backend':pipeline.detector.backend,'preview_fps':round((video_manager.jpeg_id-first)/elapsed,1),
            'inference_fps':round(len(processed)/elapsed,1),'preview_age_ms':round((time.time()-video_manager.preview_encoded_at)*1000,1),
            'inference_ms':round(pipeline.inference_ms,1),'no_server_started':True}
    assert result['preview_fps']>12,'Preview remains tied to slow inference'
    (ROOT/'artifacts/preview-benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
finally:
    stopped.set();thread.join(timeout=5);video_manager.stop();pipeline.executor.shutdown(wait=True)
