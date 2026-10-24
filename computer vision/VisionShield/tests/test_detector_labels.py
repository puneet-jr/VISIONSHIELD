import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from app.engine.detector import ObjectDetector
from app.config import config

class DetectorLabelTests(unittest.TestCase):
    def test_knife_is_enabled_using_model_names(self):
        model=MagicMock()
        model.names={0:'person',24:'backpack',26:'handbag',28:'suitcase',43:'knife',45:'bowl'}
        model.predict.return_value=[]
        with patch('ultralytics.YOLO',return_value=model), patch.object(config,'DETECT_ALL_CLASSES',False), patch.object(config,'INFERENCE_BACKEND','torch'):
            detector=ObjectDetector()
            detector.detect(np.zeros((64,64,3),dtype=np.uint8))
        self.assertIn(43,model.predict.call_args.kwargs['classes'])
        self.assertNotIn(45,model.predict.call_args.kwargs['classes'])
