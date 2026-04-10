import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

print("=" * 60)
print("PPE DETECTION SYSTEM - FULLY WORKING")
print("HELMETS & SAFETY VESTS")
print("=" * 60)

# НАСТРОЙКИ
CONFIG = {
    'helmet_confidence': 0.25,
    'vest_confidence': 0.25,
    'model_path': 'ppe_models/ppe_best.pt',  # или 'ppe_models/ppe_yolov5.pt'
    'min_helmet_size': 20,
    'min_vest_size': 40,
}

# Загружаем модель
print("\n🔄 Loading PPE model...")
try:
    model = YOLO(CONFIG['model_path'])
    print(f"✅ Model loaded!")
    print(f"   Classes: {model.names}")
except Exception as e:
    print(f"⚠️ Error: {e}")
    print("🔄 Trying alternative model...")
    model = YOLO('ppe_models/ppe_yolov5.pt')
    print(f"✅ Alternative model loaded!")
    print(f"   Classes: {model.names}")

# Открываем камеру
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🎮 Controls:")
print("   'q' - quit")
print("   's' - save screenshot")
print("   'r' - reset settings")
print("=" * 60 + "\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Детекция
    results = model(frame, conf=0.25, verbose=False)
    
    helmet_count = 0
    vest_count = 0
    
    # Обработка результатов
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            class_name = model.names[cls_id]
            
            class_lower = class_name.lower()
            
            # КАСКИ
            if any(k in class_lower for k in ['helmet', 'hardhat']):
                color = (0, 165, 255)  # Оранжевый
                label = f"HELMET {conf_val:.2f}"
                helmet_count += 1
                thickness = 3
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1 + w + 5, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # ЖИЛЕТЫ
            elif any(k in class_lower for k in ['vest', 'safety_vest']):
                color = (0, 255, 0)  # Зеленый
                label = f"VEST {conf_val:.2f}"
                vest_count += 1
                thickness = 3
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1 + w + 5, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # ЛЮДИ (опционально)
            elif 'person' in class_lower:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    # ИНФОРМАЦИОННАЯ ПАНЕЛЬ
    cv2.rectangle(frame, (10, 10), (450, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (450, 120), (0, 255, 0), 2)
    
    cv2.putText(frame, "PPE DETECTION SYSTEM", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"HELMETS: {helmet_count}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, f"VESTS: {vest_count}", (20, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    total_ppe = helmet_count + vest_count
    cv2.putText(frame, f"TOTAL PPE: {total_ppe}", (20, 115), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if total_ppe > 0 else (0, 0, 255), 1)
    
    cv2.imshow('PPE Detection System - WORKING!', frame)
    
    # Управление
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"ppe_detection_{frame_count}_H{helmet_count}_V{vest_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Finished! Processed {frame_count} frames")
print(f"   Final detection: Helmets: {helmet_count}, Vests: {vest_count}")