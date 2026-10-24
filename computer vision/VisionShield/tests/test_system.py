import asyncio
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.config import config
from app.main import app
from app.core.event_hub import EventHub
from app.engine.pipeline import SurveillancePipeline
from app.ingestion.video_stream import video_manager


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_fanout_and_slow_subscriber(self):
        hub = EventHub()
        hub.bind(asyncio.get_running_loop())
        queue = hub.subscribe()
        thread = threading.Thread(target=lambda: hub.publish({"type": "worker"}))
        thread.start()
        thread.join()
        self.assertEqual((await asyncio.wait_for(queue.get(), 1))["type"], "worker")
        for n in range(120):
            await hub.broadcast({"n": n})
        self.assertEqual(queue.qsize(), 100)
        self.assertEqual((await queue.get())["n"], 20)
        hub.unsubscribe(queue)


class ReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_alert_reports_off_inference_thread(self):
        from app.core.event_hub import event_hub
        hub = event_hub
        hub.bind(asyncio.get_running_loop())
        queue = hub.subscribe()
        pipeline = SurveillancePipeline()
        detection = {"box": [400, 150, 480, 280], "centroid": (440, 215), "confidence": 0.9, "class_id": 0, "label": "person", "aspect_ratio": 1.625}
        report_threads = []
        def report(event, frame):
            report_threads.append(threading.current_thread().name)
            return {"incident_title": "Test report"}
        try:
            video_manager.start('phone_upload')
            video_manager.push_phone_frame(np.zeros((480, 640, 3), dtype=np.uint8))
            with patch.object(pipeline.detector, 'detect', return_value=[detection]), patch.object(pipeline.geofence, 'process_geofence', return_value=[{'alert_type':'PERIMETER_BREACH','severity':'HIGH','track_id':1,'object_label':'person'}]), patch('app.engine.pipeline.db.insert_event'), patch('app.engine.pipeline.incident_dispatcher.generate_dossier', side_effect=report):
                await asyncio.to_thread(pipeline.run_cycle)
                first = await asyncio.wait_for(queue.get(), 2)
                second = await asyncio.wait_for(queue.get(), 2)
                self.assertEqual({first['type'], second['type']}, {'SECURITY_EVENT', 'TACTICAL_INCIDENT'})
                self.assertTrue(report_threads[0].startswith('incident'))
        finally:
            hub.unsubscribe(queue)
            pipeline.executor.shutdown(wait=True)
            video_manager.stop()


class ApiTests(unittest.TestCase):
    def test_pages_pairing_and_input_validation(self):
        client = TestClient(app)
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.get('/api/qr').headers['content-type'], 'image/png')
        self.assertEqual(client.get('/api/events?limit=-1').status_code, 422)
        self.assertEqual(client.post('/api/zones', json={"zones": [{"id": "x", "name": "bad", "points": [[2, 0], [0, 0], [0, 1]]}]}).status_code, 422)
        remote = TestClient(app, base_url='https://visionshield.remote')
        self.assertEqual(remote.get('/api/info').status_code, 401)
        self.assertEqual(remote.get('/phone').status_code, 200)
        self.assertEqual(remote.post('/api/pair', json={"token": "bad"}).status_code, 403)
        self.assertEqual(remote.post('/api/pair', json={"token": config.ACCESS_TOKEN}).status_code, 200)
        self.assertEqual(remote.get('/api/info').status_code, 200)
        self.assertEqual(remote.post('/api/upload_frame', files={"file": ("bad.jpg", b'bad')}).status_code, 422)

    def test_camera_websocket_frame_acknowledgement(self):
        client = TestClient(app)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        ok, jpeg = cv2.imencode('.jpg', frame)
        self.assertTrue(ok)
        with client.websocket_connect('/ws/camera') as ws:
            ws.send_bytes(jpeg.tobytes())
            self.assertTrue(ws.receive_json()['ok'])
            self.assertEqual(video_manager.source_type, 'phone_upload')
            self.assertEqual(video_manager.get_raw_frame().shape, frame.shape)
        video_manager.stop()

    def test_source_change_stops_existing_phone_sender(self):
        from starlette.websockets import WebSocketDisconnect
        client=TestClient(app)
        _,jpeg=cv2.imencode('.jpg',np.zeros((120,160,3),dtype=np.uint8))
        try:
            with client.websocket_connect('/ws/camera') as ws:
                ws.send_bytes(jpeg.tobytes());self.assertTrue(ws.receive_json()['ok'])
                self.assertEqual(client.post('/api/source',json={'source':'phone_upload'}).status_code,200)
                ws.send_bytes(jpeg.tobytes())
                with self.assertRaises(WebSocketDisconnect) as closed:
                    ws.receive_json()
                self.assertEqual(closed.exception.code,4000)
        finally:
            video_manager.stop()

    def test_stale_or_previous_camera_boxes_are_hidden(self):
        from app.engine.pipeline import pipeline
        import time
        client=TestClient(app)
        old=pipeline.snapshot
        try:
            pipeline.snapshot={'generation':video_manager.generation-1,'captured_at':time.time(),'tracks':[{'track_id':99}]}
            payload=client.get('/api/telemetry').json()
            self.assertTrue(payload['stale']);self.assertEqual(payload['tracks'],[])
        finally:
            pipeline.snapshot=old

    def test_inference_processes_each_frame_once(self):
        pipeline = SurveillancePipeline()
        try:
            video_manager.start('phone_upload')
            video_manager.push_phone_frame(np.zeros((240, 320, 3), dtype=np.uint8))
            with patch.object(pipeline.detector, 'detect', return_value=[]) as detect:
                pipeline.run_cycle()
                pipeline.run_cycle()
                self.assertEqual(detect.call_count, 1)
                import time
                deadline=time.monotonic()+1
                while video_manager.jpeg is None and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertIsNotNone(video_manager.jpeg)
        finally:
            video_manager.stop()
            pipeline.executor.shutdown(wait=False)


if __name__ == '__main__':
    unittest.main()
