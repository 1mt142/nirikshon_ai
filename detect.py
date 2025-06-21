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
import geocoder
from datetime import datetime

load_dotenv()

# Load configuration
TARGET_CLASS = int(os.getenv('TARGET_CLASS', 67))  # 67 for cell phone
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.6))
DETECTION_COOLDOWN = float(os.getenv('DETECTION_COOLDOWN', 100))
WHATSAPP_NUMBER = os.getenv('WHATSAPP_NUMBER')
WHATSAPP_MESSAGE = os.getenv('WHATSAPP_MESSAGE', '🚨 Alert: Object Detected!')
ENABLE_GEOLOCATION = os.getenv('ENABLE_GEOLOCATION', 'False').lower() == 'true'

model = YOLO("yolov8n.pt")

use_ip_camera = os.getenv('USE_IP_CAMERA', 'False').lower() == 'true'
ip_camera_url = os.getenv('IP_CAMERA_URL', None)
if use_ip_camera and ip_camera_url:
    cap = cv2.VideoCapture(ip_camera_url)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

os.makedirs("detections", exist_ok=True)

frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

pygame.mixer.init()

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

# Cache location
last_location = None
last_location_time = 0
LOCATION_REFRESH_INTERVAL = 300  # 5 minutes

# Object tracking
recent_detections = []  
MAX_RECENT_TIME = 60  # 1 minute
POSITION_THRESHOLD = 50  # pixels

def is_same_object(new_box, recent_entries):
    x1, y1, x2, y2 = new_box
    new_cx, new_cy = (x1 + x2) / 2, (y1 + y2) / 2
    now = time.time()
    
    for entry in recent_entries:
        old_time, old_box, _ = entry
        if now - old_time > MAX_RECENT_TIME:
            continue
        ox1, oy1, ox2, oy2 = old_box
        ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        dist = np.sqrt((new_cx - ocx) ** 2 + (new_cy - ocy) ** 2)
        if dist < POSITION_THRESHOLD:
            return True
    return False

def get_current_location():
    if not ENABLE_GEOLOCATION:
        return None
        
    global last_location, last_location_time
    try:
        current_time = time.time()
        if last_location and (current_time - last_location_time < LOCATION_REFRESH_INTERVAL):
            return last_location
        g = geocoder.ip('me')
        if g.latlng:
            last_location = g.latlng
            last_location_time = current_time
            return last_location
    except Exception as e:
        print(f"[GPS Error] {e}")
    return None

def detection_worker():
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
    try:
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[Sound Error] {e}")

def send_whatsapp_alert(count, image_path=None, location=None):
    def _send():
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            message = f"{WHATSAPP_MESSAGE}\nCount: {count}\nTime: {timestamp}"
            if location and ENABLE_GEOLOCATION:
                message += f"\nLocation: https://maps.google.com/?q={location[0]},{location[1]}"
            
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
print(f"Geolocation enabled: {ENABLE_GEOLOCATION}")
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
            
            for detection in detections:
                # Check if this is a new detection (not seen recently)
                if not is_same_object(detection['box'], recent_detections):
                    object_count += 1
                    last_detection_time = current_time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detections/detect_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    # Get location only if enabled
                    location = get_current_location() if ENABLE_GEOLOCATION else None
                    
                    play_alert_sound()
                    send_whatsapp_alert(object_count, filename, location)
                    
                    # Mark this detection as notified
                    recent_detections.append((
                        time.time(), 
                        detection['box'], 
                        True  # notification_sent flag
                    ))
                    break
                    
        # Clean up old detections
        recent_detections = [
            entry for entry in recent_detections 
            if current_time - entry[0] <= MAX_RECENT_TIME
        ]
        
        # Draw detections on frame
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['box'])
            label = f"{detection['class_name']} {detection['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display info on frame
        cv2.putText(frame, f"Detecting: {coco_classes[TARGET_CLASS]}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {object_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {1/(time.time()-last_frame_time):.1f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if ENABLE_GEOLOCATION and last_location:
            gps_text = f"GPS: {last_location[0]:.4f}, {last_location[1]:.4f}"
            cv2.putText(frame, gps_text, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
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