import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

print("=" * 60)
print("PPE DETECTION SYSTEM - HELMETS & VESTS")
print("=" * 60)

# ==================== РАСШИРЕННЫЕ НАСТРОЙКИ ====================
CONFIG = {
    # Основные настройки
    'confidence': 0.35,
    'iou': 0.45,
    'max_detections': 150,
    
    # Специальные настройки для разных типов PPE
    'helmet_confidence': 0.1,     # Порог для касок
    'vest_confidence': 0.35,      # Порог для жилетов
    'helmet_iou': 0.4,
    'vest_iou': 0.45,
    
    # Аугментация
    'augment': True,
    'multi_scale': True,
    
    # Фильтрация по размеру
    'min_helmet_size': 40,         # Минимальный размер каски (пикселей)
    'max_helmet_size': 300,
    'min_vest_size': 80,           # Минимальный размер жилета
    'max_vest_size': 500,
    
    'tracking_enabled': True,
    'tracking_frames': 5,
    
    # КЛАССЫ PPE (исправлено!)
    'helmet_classes': [
        'helmet', 'hardhat', 'construction_helmet', 'safety_helmet',
        'hard_hat', 'helmet_yellow', 'helmet_white', 'helmet_red', 
        'helmet_blue', 'helmet_orange', 'helmet_green', 'head'
    ],
    
    'vest_classes': [
        'vest', 'safety_vest', 'reflective_vest', 'high_visibility_vest',
        'vest_orange', 'vest_yellow', 'safety_jacket', 'vest_red',
        'construction_vest', 'work_vest', 'ppe_vest'
    ],
    
    # Модель
    'use_custom_model': True,
    'custom_model_path': 'ppe_best.pt',
    
    # Визуализация
    'show_confidence': True,
    'show_tracking': True,
    'heatmap_enabled': False,
}

print("\n🔧 PPE DETECTION CONFIGURATION:")
print(f"   Helmet confidence: {CONFIG['helmet_confidence']}")
print(f"   Vest confidence: {CONFIG['vest_confidence']}")
print(f"   Helmet IOU: {CONFIG['helmet_iou']}")
print(f"   Vest IOU: {CONFIG['vest_iou']}")
print(f"   Augmentation: {CONFIG['augment']}")
print(f"   Tracking: {CONFIG['tracking_enabled']}")

# Загружаем модель
print("\n🔄 Loading YOLO model...")
try:
    if CONFIG['use_custom_model']:
        model = YOLO(CONFIG['custom_model_path'])
        print(f"✅ Custom model loaded: {CONFIG['custom_model_path']}")
        print(f"   Model classes: {model.names}")
    else:
        model = YOLO("yolov8n.pt")
        print("✅ Base model loaded: yolov8n.pt")
except Exception as e:
    print(f"⚠️ Error loading custom model: {e}")
    print("🔄 Falling back to base model...")
    model = YOLO("yolov8n.pt")

# Открываем камеру
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Для трекинга
if CONFIG['tracking_enabled']:
    helmet_tracker = {}
    vest_tracker = {}
    next_helmet_id = 0
    next_vest_id = 0

def is_helmet(class_name, class_id, model_names):
    """Проверка, является ли объект каской"""
    class_name_lower = class_name.lower()
    
    for helmet_class in CONFIG['helmet_classes']:
        if helmet_class in class_name_lower:
            return True
    
    if CONFIG['use_custom_model']:
        # Часто в PPE моделях класс 0 - каска
        if class_id == 0:
            return True
    
    return False

def is_vest(class_name, class_id, model_names):
    """Проверка, является ли объект жилетом"""
    class_name_lower = class_name.lower()
    
    for vest_class in CONFIG['vest_classes']:
        if vest_class in class_name_lower:
            return True
    
    if CONFIG['use_custom_model']:
        # Часто в PPE моделях класс 1 - жилет
        if class_id == 1:
            return True
    
    return False

def calculate_ppe_confidence(box, frame_shape, ppe_type):
    """Рассчитываем дополнительную уверенность для PPE"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    frame_area = frame_shape[0] * frame_shape[1]
    size_ratio = area / frame_area
    
    # Бонус за размер в зависимости от типа PPE
    size_bonus = 1.0
    if ppe_type == 'helmet':
        if 0.005 < size_ratio < 0.05:
            size_bonus = 1.2
        elif size_ratio > 0.1:
            size_bonus = 0.8
    else:  # vest
        if 0.02 < size_ratio < 0.15:
            size_bonus = 1.2
        elif size_ratio > 0.25:
            size_bonus = 0.8
    
    # Бонус за пропорции
    aspect_ratio = width / height
    aspect_bonus = 1.0
    if ppe_type == 'helmet':
        if 0.8 < aspect_ratio < 1.3:
            aspect_bonus = 1.15
    else:  # vest
        if 0.6 < aspect_ratio < 1.0:
            aspect_bonus = 1.15
    
    return size_bonus * aspect_bonus

# Переменные для статистики
frame_count = 0
helmet_detections = deque(maxlen=30)
vest_detections = deque(maxlen=30)
helmet_confidence = CONFIG['helmet_confidence']
vest_confidence = CONFIG['vest_confidence']

print("\n🎮 Controls:")
print("   'q' - quit")
print("   's' - save screenshot")
print("   '1' / '2' - adjust helmet/vest confidence")
print("   '3' / '4' - adjust helmet/vest IOU")
print("   't' - toggle tracking")
print("   'h' - show/hide heatmap")
print("   'r' - reset to default settings")
print("\n🔍 Detecting: HELMETS and SAFETY VESTS")
print("=" * 60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    h, w = frame.shape[:2]
    
    # Детекция
    results = model(frame, conf=0.25, iou=0.45, max_det=CONFIG['max_detections'], 
                    augment=CONFIG['augment'], verbose=False)
    
    # Списки обнаруженных объектов
    current_helmets = []
    current_vests = []
    
    # Обработка результатов
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            class_name = model.names[cls_id]
            
            # Проверяем каски
            if is_helmet(class_name, cls_id, model.names):
                helmet_width = x2 - x1
                helmet_height = y2 - y1
                
                if (helmet_width > CONFIG['min_helmet_size'] and 
                    helmet_height > CONFIG['min_helmet_size'] and
                    helmet_width < CONFIG['max_helmet_size'] and
                    helmet_height < CONFIG['max_helmet_size'] and
                    conf_val >= helmet_confidence):
                    
                    extra_confidence = calculate_ppe_confidence((x1, y1, x2, y2), frame.shape, 'helmet')
                    final_confidence = min(1.0, conf_val * extra_confidence)
                    
                    current_helmets.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': final_confidence,
                        'original_conf': conf_val,
                        'class': class_name,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'size': (helmet_width, helmet_height)
                    })
            
            # Проверяем жилеты
            elif is_vest(class_name, cls_id, model.names):
                vest_width = x2 - x1
                vest_height = y2 - y1
                
                if (vest_width > CONFIG['min_vest_size'] and 
                    vest_height > CONFIG['min_vest_size'] and
                    vest_width < CONFIG['max_vest_size'] and
                    vest_height < CONFIG['max_vest_size'] and
                    conf_val >= vest_confidence):
                    
                    extra_confidence = calculate_ppe_confidence((x1, y1, x2, y2), frame.shape, 'vest')
                    final_confidence = min(1.0, conf_val * extra_confidence)
                    
                    current_vests.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': final_confidence,
                        'original_conf': conf_val,
                        'class': class_name,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'size': (vest_width, vest_height)
                    })
    
    # Трекинг для касок
    if CONFIG['tracking_enabled']:
        # ... (аналогичный код трекинга для касок и жилетов)
        pass
    
    # Статистика
    helmet_detections.append(len(current_helmets))
    vest_detections.append(len(current_vests))
    avg_helmets = np.mean(helmet_detections) if helmet_detections else 0
    avg_vests = np.mean(vest_detections) if vest_detections else 0
    
    # ВИЗУАЛИЗАЦИЯ
    # Рисуем каски (синий цвет)
    for helmet in current_helmets:
        x1, y1, x2, y2 = helmet['bbox']
        conf_val = helmet['confidence']
        
        if conf_val > 0.7:
            color = (255, 100, 0)  # Оранжевый для высокой уверенности
            thickness = 3
        elif conf_val > 0.5:
            color = (0, 200, 255)  # Желтый
            thickness = 2
        else:
            color = (255, 100, 100)  # Светло-синий
            thickness = 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"HELMET {conf_val:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-25), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Рисуем жилеты (зеленый цвет)
    for vest in current_vests:
        x1, y1, x2, y2 = vest['bbox']
        conf_val = vest['confidence']
        
        if conf_val > 0.7:
            color = (0, 255, 0)  # Ярко-зеленый
            thickness = 3
        elif conf_val > 0.5:
            color = (0, 200, 0)  # Зеленый
            thickness = 2
        else:
            color = (100, 255, 100)  # Светло-зеленый
            thickness = 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"VEST {conf_val:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-25), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # ИНФОРМАЦИОННАЯ ПАНЕЛЬ
    # Основная панель
    cv2.rectangle(frame, (10, 10), (480, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (480, 180), (0, 255, 0), 2)
    
    cv2.putText(frame, "PPE DETECTION SYSTEM", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Статистика касок
    helmet_count = len(current_helmets)
    cv2.putText(frame, f"HELMETS: {helmet_count}", 
               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
               (0, 165, 255) if helmet_count > 0 else (0, 0, 255), 2)
    
    # Статистика жилетов
    vest_count = len(current_vests)
    cv2.putText(frame, f"VESTS: {vest_count}", 
               (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
               (0, 255, 0) if vest_count > 0 else (0, 0, 255), 2)
    
    cv2.putText(frame, f"AVG HELMETS: {avg_helmets:.1f} | AVG VESTS: {avg_vests:.1f}", 
               (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    total_ppe = helmet_count + vest_count
    cv2.putText(frame, f"TOTAL PPE: {total_ppe}", 
               (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 
               (0, 255, 0) if total_ppe > 0 else (0, 0, 255), 1)
    
    # Панель настроек
    cv2.rectangle(frame, (frame.shape[1]-300, 10), (frame.shape[1]-10, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (frame.shape[1]-300, 10), (frame.shape[1]-10, 180), (150, 150, 150), 1)
    
    cv2.putText(frame, "PPE SETTINGS", (frame.shape[1]-290, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Helmet conf: {helmet_confidence:.2f} (1/2)", 
               (frame.shape[1]-290, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    cv2.putText(frame, f"Vest conf: {vest_confidence:.2f} (3/4)", 
               (frame.shape[1]-290, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(frame, f"Tracking: {'ON' if CONFIG['tracking_enabled'] else 'OFF'} (t)", 
               (frame.shape[1]-290, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
               (0, 255, 0) if CONFIG['tracking_enabled'] else (0, 0, 255), 1)
    cv2.putText(frame, f"Augment: {'ON' if CONFIG['augment'] else 'OFF'}", 
               (frame.shape[1]-290, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
               (0, 255, 0) if CONFIG['augment'] else (0, 0, 255), 1)
    cv2.putText(frame, f"Detections: H:{helmet_count} V:{vest_count}", 
               (frame.shape[1]-290, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
               (150, 150, 150), 1)
    cv2.putText(frame, f"Total PPE: {total_ppe}", 
               (frame.shape[1]-290, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
               (255, 255, 0), 1)
    
    # FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-100, frame.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.imshow('PPE Detection System - Helmets & Vests', frame)
    
    # Управление
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"ppe_detection_{frame_count}_H{helmet_count}_V{vest_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    # Регулировка порогов
    elif key == ord('1'):
        helmet_confidence = min(0.9, helmet_confidence + 0.05)
        print(f"🔧 Helmet confidence: {helmet_confidence:.2f}")
    elif key == ord('2'):
        helmet_confidence = max(0.2, helmet_confidence - 0.05)
        print(f"🔧 Helmet confidence: {helmet_confidence:.2f}")
    elif key == ord('3'):
        vest_confidence = min(0.9, vest_confidence + 0.05)
        print(f"🔧 Vest confidence: {vest_confidence:.2f}")
    elif key == ord('4'):
        vest_confidence = max(0.2, vest_confidence - 0.05)
        print(f"🔧 Vest confidence: {vest_confidence:.2f}")
    
    elif key == ord('t'):
        CONFIG['tracking_enabled'] = not CONFIG['tracking_enabled']
        print(f"🔄 Tracking: {'ON' if CONFIG['tracking_enabled'] else 'OFF'}")
    
    elif key == ord('r'):
        helmet_confidence = CONFIG['helmet_confidence']
        vest_confidence = CONFIG['vest_confidence']
        print(f"🔄 Settings reset: Helmet={helmet_confidence:.2f}, Vest={vest_confidence:.2f}")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Finished! Processed {frame_count} frames")