import cv2
from ultralytics import YOLO
import numpy as np
import os
import time
import threading
import winsound

print("=" * 60)
print("PPE MONITORING SYSTEM WITH VIOLATION DETECTION")
print("=" * 60)

# Список моделей для тестирования
models_to_test = [
    ('ppe_models/ppe_yolov5.pt', 'YOLOv5 PPE (alisobhy22)'),
    ('ppe_models/ppe_best.pt', 'YOLOv8 PPE (mridulchdry17)'),
]

# Пробуем загрузить каждую модель
loaded_models = []
for model_path, model_name in models_to_test:
    try:
        print(f"\n🔄 Loading {model_name}...")
        print(f"   Path: {model_path}")
        
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ SUCCESS! Model loaded: {model_name}")
            print(f"   Classes in this model: {model.names}")
            loaded_models.append((model, model_name, model_path))
        else:
            print(f"❌ File not found: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")

if not loaded_models:
    print("\n❌ No models could be loaded! Check if files exist in ppe_models folder.")
    exit()

print(f"\n✅ Loaded {len(loaded_models)} model(s)")
print("=" * 60)

# Выбираем первую модель для начала
current_model_idx = 0
model, model_name, model_path = loaded_models[current_model_idx]
print(f"\n🔴 USING MODEL: {model_name}")

# НАСТРОЙКИ ПО УМОЛЧАНИЮ
detect_helmet = True
detect_vest = True
detect_mask = False

# ПОРОГИ ДЛЯ РАЗНЫХ КЛАССОВ
CONFIDENCE_THRESHOLDS = {
    'helmet': 0.25,
    'vest': 0.25,
    'mask': 0.25,
    'person': 0.30
}

# НАСТРОЙКИ ВРЕМЕНИ И НАРУШЕНИЙ
VIOLATION_TIME_SEC = 10  # секунд без защиты до нарушения
ALARM_DURATION_SEC = 2   # длительность сигнала тревоги
MAX_VIOLATIONS = 100     # максимальное количество нарушений для сохранения

# Словари для отслеживания времени нарушений
helmet_violation_start = None  # время начала отсутствия каски
vest_violation_start = None    # время начала отсутствия жилета
mask_violation_start = None    # время начала отсутствия маски

violation_log = []  # список нарушений
current_alarm = False
alarm_end_time = 0

# Создаем папку для скриншотов нарушений
violations_folder = "violations"
if not os.path.exists(violations_folder):
    os.makedirs(violations_folder)
    print(f"📁 Created folder: {violations_folder}")
else:
    print(f"📁 Using existing folder: {violations_folder}")

def play_alarm():
    """Воспроизводит звуковой сигнал тревоги"""
    global current_alarm, alarm_end_time
    
    if current_alarm:
        return
    
    current_alarm = True
    alarm_end_time = time.time() + ALARM_DURATION_SEC
    
    def alarm_sound():
        try:
            for _ in range(2):
                winsound.Beep(1000, 500)
                time.sleep(0.1)
        except:
            print("\n🔔 ALARM! 🔔")
        
        global current_alarm
        current_alarm = False
    
    threading.Thread(target=alarm_sound, daemon=True).start()

def log_violation(ppe_type, frame, person_count):
    """Логирует нарушение и сохраняет скриншот"""
    global violation_log
    
    # Проверяем лимит нарушений
    if len(violation_log) >= MAX_VIOLATIONS:
        print(f"⚠️ Maximum violations ({MAX_VIOLATIONS}) reached! Not saving more.")
        return
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{violations_folder}/violation_{ppe_type}_{timestamp}.jpg"
    
    cv2.imwrite(filename, frame)
    
    violation = {
        'time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'type': ppe_type,
        'screenshot': filename,
        'people_count': person_count
    }
    violation_log.append(violation)
    
    print(f"\n⚠️ VIOLATION DETECTED! ⚠️")
    print(f"   Type: NO {ppe_type.upper()}")
    print(f"   Time: {violation['time']}")
    print(f"   People on screen: {person_count}")
    print(f"   Screenshot: {filename}")
    print(f"   Total violations: {len(violation_log)}")
    
    play_alarm()

def get_dynamic_threshold(class_name, cls_id):
    """Возвращает порог уверенности в зависимости от класса"""
    class_lower = class_name.lower()
    
    if cls_id in [2, 5] or 'hardhat' in class_lower or 'helmet' in class_lower:
        return CONFIDENCE_THRESHOLDS['helmet']
    elif cls_id in [7, 11] or 'vest' in class_lower:
        return CONFIDENCE_THRESHOLDS['vest']
    elif cls_id in [4, 6] or 'mask' in class_lower:
        return CONFIDENCE_THRESHOLDS['mask']
    elif cls_id == 8 or 'person' in class_lower:
        return CONFIDENCE_THRESHOLDS['person']
    else:
        return 0.25

def draw_menu(frame, detect_helmet, detect_vest, detect_mask, model_name, violation_count):
    h, w = frame.shape[:2]
    
    # Основное меню (левая панель)
    cv2.rectangle(frame, (10, 10), (500, 290), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (500, 290), (255, 255, 255), 2)
    
    cv2.putText(frame, "PPE MONITORING SYSTEM", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press number to toggle:", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    color1 = (0, 255, 0) if detect_helmet else (0, 0, 255)
    status1 = "ON" if detect_helmet else "OFF"
    cv2.putText(frame, f"1 - HELMET     [{status1}]", (20, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, color1, 2)
    
    color2 = (0, 255, 0) if detect_vest else (0, 0, 255)
    status2 = "ON" if detect_vest else "OFF"
    cv2.putText(frame, f"2 - VEST       [{status2}]", (20, 125), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, color2, 2)
    
    color3 = (0, 255, 0) if detect_mask else (0, 0, 255)
    status3 = "ON" if detect_mask else "OFF"
    cv2.putText(frame, f"3 - MASK       [{status3}]", (20, 155), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, color3, 2)
    
    cv2.putText(frame, "-" * 45, (20, 185), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(frame, f"VIOLATIONS: {violation_count}", (20, 210), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"ALARM: {'ACTIVE' if current_alarm else 'OK'}", (20, 235), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if current_alarm else (0, 255, 0), 1)
    cv2.putText(frame, f"Threshold: {VIOLATION_TIME_SEC}s", (20, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, " +/- to adjust time", (20, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # Правая панель - текущая модель
    cv2.rectangle(frame, (w-320, 10), (w-10, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (w-320, 10), (w-10, 100), (255, 255, 255), 1)
    cv2.putText(frame, "CURRENT MODEL:", (w-310, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    model_short = model_name[:25] if len(model_name) > 25 else model_name
    cv2.putText(frame, model_short, (w-310, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    active = []
    if detect_helmet: active.append("HELMET")
    if detect_vest: active.append("VEST")
    if detect_mask: active.append("MASK")
    
    active_text = ", ".join(active) if active else "NONE"
    cv2.putText(frame, f"ACTIVE: {active_text}", (w-310, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Открываем камеру
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("\n🎮 CONTROLS:")
print("   1 - Toggle HELMET detection")
print("   2 - Toggle VEST detection")
print("   3 - Toggle MASK detection")
print("   4 - Switch model")
print("   + - Increase violation threshold (+5 sec)")
print("   - - Decrease violation threshold (-5 sec)")
print("   r - Reset violation log")
print("   s - Save screenshot")
print("   q - Quit")
print(f"\n⚙️ RULES:")
print(f"   - Violation counted ONLY if person is on screen")
print(f"   - Missing PPE > {VIOLATION_TIME_SEC}s = VIOLATION + screenshot + alarm")
print(f"   - No person on screen = no timer, no violation")
print(f"   - Alarm duration: {ALARM_DURATION_SEC} seconds")
print(f"   - Max violations stored: {MAX_VIOLATIONS}")
print(f"   - Violations saved to: {violations_folder}/")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue
    
    frame = cv2.flip(frame, 1)
    
    # Детекция
    min_conf = min(CONFIDENCE_THRESHOLDS.values())
    results = model(frame, conf=min_conf, iou=0.45, verbose=False)
    
    # Счетчики
    helmet_count = 0
    vest_count = 0
    mask_count = 0
    person_count = 0
    no_helmet_count = 0
    no_vest_count = 0
    no_mask_count = 0
    
    # Обработка результатов
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            class_name = model.names[cls_id]
            class_lower = class_name.lower()
            
            required_conf = get_dynamic_threshold(class_name, cls_id)
            if conf_val < required_conf:
                continue
            
            # КАСКИ
            if detect_helmet and (cls_id == 2 or cls_id == 5 or 'hardhat' in class_lower or 'helmet' in class_lower):
                if cls_id == 5 or 'no-hardhat' in class_lower:
                    color = (0, 0, 255)
                    label = f"NO HELMET! {class_name} {conf_val:.2f}"
                    no_helmet_count += 1
                else:
                    color = (0, 165, 255)
                    label = f"HELMET: {class_name} {conf_val:.2f}"
                    helmet_count += 1
                thickness = 3
            
            # ЖИЛЕТЫ
            elif detect_vest and (cls_id == 11 or cls_id == 7 or 'vest' in class_lower):
                if cls_id == 7 or 'no-safety vest' in class_lower:
                    color = (0, 0, 255)
                    label = f"NO VEST! {class_name} {conf_val:.2f}"
                    no_vest_count += 1
                else:
                    color = (0, 255, 0)
                    label = f"VEST: {class_name} {conf_val:.2f}"
                    vest_count += 1
                thickness = 3
            
            # МАСКИ
            elif detect_mask and (cls_id == 4 or cls_id == 6 or 'mask' in class_lower):
                if cls_id == 6 or 'no-mask' in class_lower:
                    color = (0, 0, 255)
                    label = f"NO MASK! {class_name} {conf_val:.2f}"
                    no_mask_count += 1
                else:
                    color = (255, 255, 0)
                    label = f"MASK: {class_name} {conf_val:.2f}"
                    mask_count += 1
                thickness = 3
            
            # ЛЮДИ
            elif cls_id == 8 or 'person' in class_lower:
                color = (255, 255, 255)
                label = f"PERSON: {conf_val:.2f}"
                person_count += 1
                thickness = 1
            else:
                continue
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-22), (x1 + label_w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1-6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # ЛОГИКА НАРУШЕНИЙ 
    # Нарушение считается ТОЛЬКО если есть люди на экране
    current_time = time.time()
    
    # ПРОВЕРКА КАСОК
    if detect_helmet:
        if person_count > 0:
            has_helmet = (helmet_count > 0)
            
            if not has_helmet:
                if helmet_violation_start is None:
                    helmet_violation_start = current_time
                    print(f"⏱️ No helmet detected! Timer started...")
                else:
                    elapsed = current_time - helmet_violation_start
                    if elapsed >= VIOLATION_TIME_SEC:
                        log_violation("HELMET", frame, person_count)
                        helmet_violation_start = current_time
            else:
                if helmet_violation_start is not None:
                    print(f"✅ Helmet detected! Timer reset.")
                    helmet_violation_start = None
        else:
            if helmet_violation_start is not None:
                print(f"👤 No people on screen, helmet timer reset.")
                helmet_violation_start = None
    
    # ПРОВЕРКА ЖИЛЕТОВ
    if detect_vest:
        if person_count > 0:
            has_vest = (vest_count > 0)
            
            if not has_vest:
                if vest_violation_start is None:
                    vest_violation_start = current_time
                    print(f"⏱️ No vest detected! Timer started...")
                else:
                    elapsed = current_time - vest_violation_start
                    if elapsed >= VIOLATION_TIME_SEC:
                        log_violation("VEST", frame, person_count)
                        vest_violation_start = current_time
            else:
                if vest_violation_start is not None:
                    print(f"✅ Vest detected! Timer reset.")
                    vest_violation_start = None
        else:
            if vest_violation_start is not None:
                print(f"👤 No people on screen, vest timer reset.")
                vest_violation_start = None
    
    # ПРОВЕРКА МАСОК
    if detect_mask:
        if person_count > 0:
            has_mask = (mask_count > 0)
            
            if not has_mask:
                if mask_violation_start is None:
                    mask_violation_start = current_time
                    print(f"⏱️ No mask detected! Timer started...")
                else:
                    elapsed = current_time - mask_violation_start
                    if elapsed >= VIOLATION_TIME_SEC:
                        log_violation("MASK", frame, person_count)
                        mask_violation_start = current_time
            else:
                if mask_violation_start is not None:
                    print(f"✅ Mask detected! Timer reset.")
                    mask_violation_start = None
        else:
            if mask_violation_start is not None:
                print(f"👤 No people on screen, mask timer reset.")
                mask_violation_start = None
    
    # Проверяем, закончился ли сигнал тревоги
    if current_alarm and time.time() > alarm_end_time:
        current_alarm = False
    
    # СТАТИСТИЧЕСКАЯ ПАНЕЛЬ (правая сторона)
    h, w = frame.shape[:2]
    stats_y = 120
    
    cv2.rectangle(frame, (w-320, stats_y), (w-10, stats_y + 250), (0, 0, 0), -1)
    cv2.rectangle(frame, (w-320, stats_y), (w-10, stats_y + 250), (0, 255, 0), 1)
    cv2.putText(frame, "STATUS & TIMERS", (w-310, stats_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    y_offset = stats_y + 50
    
    # Статус касок
    if detect_helmet:
        if person_count == 0:
            cv2.putText(frame, "HELMET: NO PEOPLE", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        elif helmet_count > 0:
            cv2.putText(frame, "HELMET: PROTECTED ✅", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        elif helmet_violation_start:
            elapsed = int(current_time - helmet_violation_start)
            remaining = max(0, VIOLATION_TIME_SEC - elapsed)
            cv2.putText(frame, f"HELMET: MISSING! ({remaining}s)", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "HELMET: OK", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y_offset += 30
    
    # Статус жилетов
    if detect_vest:
        if person_count == 0:
            cv2.putText(frame, "VEST: NO PEOPLE", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        elif vest_count > 0:
            cv2.putText(frame, "VEST: PROTECTED ✅", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        elif vest_violation_start:
            elapsed = int(current_time - vest_violation_start)
            remaining = max(0, VIOLATION_TIME_SEC - elapsed)
            cv2.putText(frame, f"VEST: MISSING! ({remaining}s)", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "VEST: OK", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y_offset += 30
    
    # Статус масок
    if detect_mask:
        if person_count == 0:
            cv2.putText(frame, "MASK: NO PEOPLE", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        elif mask_count > 0:
            cv2.putText(frame, "MASK: PROTECTED ✅", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        elif mask_violation_start:
            elapsed = int(current_time - mask_violation_start)
            remaining = max(0, VIOLATION_TIME_SEC - elapsed)
            cv2.putText(frame, f"MASK: MISSING! ({remaining}s)", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "MASK: OK", (w-310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y_offset += 30
    
    cv2.putText(frame, "-" * 35, (w-310, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    y_offset += 20
    
    cv2.putText(frame, f"People on screen: {person_count}", (w-310, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_offset += 25
    
    cv2.putText(frame, f"Total violations: {len(violation_log)}", (w-310, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if len(violation_log) > 0 else (200, 200, 200), 1)
    
    # Сигнал тревоги (мигающая красная рамка)
    if current_alarm:
        if int(time.time() * 2) % 2:  # Мигание каждые 0.5 секунды
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
            cv2.putText(frame, "ALARM! MISSING PPE!", (w//2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    # Рисуем меню
    draw_menu(frame, detect_helmet, detect_vest, detect_mask, model_name, len(violation_log))
    
    cv2.imshow('PPE Monitoring System - Smart Violation Detection', frame)
    
    # Управление
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('1'):
        detect_helmet = not detect_helmet
        if not detect_helmet:
            helmet_violation_start = None
        print(f"\n🪖 HELMET detection: {'ON' if detect_helmet else 'OFF'}")
    elif key == ord('2'):
        detect_vest = not detect_vest
        if not detect_vest:
            vest_violation_start = None
        print(f"\n🦺 VEST detection: {'ON' if detect_vest else 'OFF'}")
    elif key == ord('3'):
        detect_mask = not detect_mask
        if not detect_mask:
            mask_violation_start = None
        print(f"\n😷 MASK detection: {'ON' if detect_mask else 'OFF'}")
    elif key == ord('4'):
        if len(loaded_models) > 1:
            current_model_idx = (current_model_idx + 1) % len(loaded_models)
            model, model_name, model_path = loaded_models[current_model_idx]
            print(f"\n🔄 SWITCHED TO: {model_name}")
            helmet_violation_start = None
            vest_violation_start = None
            mask_violation_start = None
    elif key == ord('+') or key == ord('='):
        VIOLATION_TIME_SEC = min(60, VIOLATION_TIME_SEC + 5)
        print(f"\n⏱️ Violation threshold increased to: {VIOLATION_TIME_SEC}s")
    elif key == ord('-') or key == ord('_'):
        VIOLATION_TIME_SEC = max(3, VIOLATION_TIME_SEC - 5)
        print(f"\n⏱️ Violation threshold decreased to: {VIOLATION_TIME_SEC}s")
    elif key == ord('r'):
        violation_log.clear()
        print(f"\n📋 Violation log cleared! Total: {len(violation_log)}")
    elif key == ord('s'):
        filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(f"📊 Total violations detected: {len(violation_log)}")
if violation_log:
    print("\n📋 VIOLATION LOG:")
    for i, v in enumerate(violation_log, 1):
        print(f"   {i}. {v['time']} - NO {v['type']} - People: {v['people_count']}")
        print(f"      📸 {v['screenshot']}")
    
    # Сохраняем отчет в файл
    report_filename = f"violations_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PPE VIOLATIONS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total violations: {len(violation_log)}\n")
        f.write(f"Threshold: {VIOLATION_TIME_SEC} seconds\n\n")
        f.write("VIOLATIONS LIST:\n")
        f.write("-" * 60 + "\n")
        for i, v in enumerate(violation_log, 1):
            f.write(f"{i}. {v['time']} - NO {v['type']}\n")
            f.write(f"   People: {v['people_count']}\n")
            f.write(f"   Screenshot: {v['screenshot']}\n\n")
    print(f"\n📄 Report saved to: {report_filename}")
else:
    print("\n✅ No violations detected. Great job!")
print(f"\n✅ Monitoring finished! Violations saved to: {violations_folder}/")
print("=" * 60)