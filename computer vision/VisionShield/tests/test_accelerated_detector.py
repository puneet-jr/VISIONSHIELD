import unittest
import numpy as np
from app.config import config
from app.engine.accelerated_detector import AcceleratedYOLO

class DecoderTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_and_stride(self):
        image,ratio,offset=AcceleratedYOLO.prepare(np.zeros((480,640,3),dtype=np.uint8),640)
        self.assertEqual(image.shape,(1,3,480,640));self.assertEqual(ratio,1);self.assertEqual(offset,(0,0))

    def test_class_aware_nms_and_coordinate_mapping(self):
        model=object.__new__(AcceleratedYOLO);model.config=config;model.names={0:'person',1:'knife'}
        predictions=np.zeros((1,6,3),dtype=np.float32)
        predictions[0,:4,:]=np.array([[50,50,50],[50,50,50],[40,40,40],[40,40,40]])
        predictions[0,4,:]=[.9,.8,.1];predictions[0,5,:]=[.1,.1,.85]
        detections=model.decode(predictions,(100,100),1,(0,0),[0,1])
        self.assertEqual({d['label'] for d in detections},{'person','knife'})
        self.assertEqual(len(detections),2)
        self.assertEqual(detections[0]['box'],[30,30,70,70])
