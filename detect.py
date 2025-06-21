import cv2
from ultralytics import YOLO
import numpy as np
import os
import time
import threading
import queue
import pywhatkit
from dotenv import load_dotenv
import pygame
from PIL import Image

load_dotenv()

TARGET_CLASS = int(os.getenv('TARGET_CLASS', 67))  # 67 for cell phone detection in test purpose
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.6))
DETECTION_COOLDOWN = float(os.getenv('DETECTION_COOLDOWN', 5.0))
WHATSAPP_NUMBER = os.getenv('WHATSAPP_NUMBER')
WHATSAPP_MESSAGE = os.getenv('WHATSAPP_MESSAGE', '🚨 Alert: Object Detected!')

model = YOLO("yolov8n.pt")

use_ip_camera = os.getenv('USE_IP_CAMERA', None)
ip_camera_url = os.getenv('IP_CAMERA_URL', None)
if use_ip_camera==True and ip_camera_url:
    cap = cv2.VideoCapture(ip_camera_url)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

os.makedirs("detections", exist_ok=True)

# queue for frame processing
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

pygame.mixer.init()

# COCO class names
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detection_worker():
    """Worker thread for object detection"""
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
            
        results = model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                if int(cls) == TARGET_CLASS and conf >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        'box': box,
                        'conf': conf,
                        'class_name': coco_classes[int(cls)]
                    })
        
        result_queue.put(detections)

def play_alert_sound():
    """Non-blocking sound alert"""
    try:
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[Sound Error] {e}")

def send_whatsapp_alert(count, image_path=None):
    """Send alert with optional image in background thread"""
    def _send():
        try:
            message = f"{WHATSAPP_MESSAGE}\nCount: {count}\nTime: {time.strftime('%H:%M:%S')}"
            
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                compressed_path = "detections/compressed.jpg"
                img.save(compressed_path, "JPEG", quality=70)
                
                pywhatkit.sendwhats_image(
                    WHATSAPP_NUMBER,
                    compressed_path,
                    message,
                    wait_time=15,
                    tab_close=True
                )
            else:
                pywhatkit.sendwhatmsg_instantly(
                    WHATSAPP_NUMBER,
                    message,
                    wait_time=15,
                    tab_close=True
                )
        except Exception as e:
            print(f"[WhatsApp Error] {e}")
    
    threading.Thread(target=_send, daemon=True).start()


detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()


object_count = 0
last_detection_time = 0
last_frame_time = time.time()
fps = 30 
frame_delay = 1/fps

print(f"Detecting: {coco_classes[TARGET_CLASS]} (Class {TARGET_CLASS})")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        
        if frame_queue.empty():
            frame_queue.put(frame.copy())
        
        detections = []
        if not result_queue.empty():
            detections = result_queue.get()
            
            if detections and (current_time - last_detection_time) > DETECTION_COOLDOWN:
                object_count += 1
                last_detection_time = current_time
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detections/detect_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                
                play_alert_sound()
                send_whatsapp_alert(object_count, filename)
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['box'])
            label = f"{detection['class_name']} {detection['conf']:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Detecting: {coco_classes[TARGET_CLASS]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {object_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {1/(time.time()-last_frame_time):.1f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        last_frame_time = time.time()
        
        cv2.imshow("Live Object Detection", frame)
        
        time.sleep(max(0, frame_delay - (time.time() - last_frame_time)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    frame_queue.put(None)
    detection_thread.join()
    cap.release()
    cv2.destroyAllWindows()