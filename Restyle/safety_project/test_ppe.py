import cv2
from ultralytics import YOLO
import numpy as np

print("=" * 60)
print("PPE DETECTION - TEST WITH CUSTOM MODEL")
print("=" * 60)

# Загружаем модель
print("\n🔄 Loading custom PPE model...")
model = YOLO('ppe_best.pt')
print(f"✅ Model loaded!")
print(f"   Model classes: {model.names}")

# Открываем камеру
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🎮 Controls:")
print("   'q' - quit")
print("   's' - save screenshot")
print("=" * 60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    
    # Детекция с низким порогом confidence
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
            
            # Определяем цвет и метку в зависимости от класса
            if 'helmet' in class_name.lower() or 'hardhat' in class_name.lower() or cls_id == 0:
                color = (0, 165, 255)  # Оранжевый для касок
                label = f"HELMET: {class_name} {conf_val:.2f}"
                helmet_count += 1
            elif 'vest' in class_name.lower() or cls_id == 1:
                color = (0, 255, 0)    # Зеленый для жилетов
                label = f"VEST: {class_name} {conf_val:.2f}"
                vest_count += 1
            else:
                color = (255, 255, 0)  # Голубой для других PPE
                label = f"PPE: {class_name} {conf_val:.2f}"
            
            # Рисуем рамку
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Рисуем подпись
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-22), (x1 + label_w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1-6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Информационная панель
    cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (350, 100), (0, 255, 0), 2)
    
    cv2.putText(frame, "PPE DETECTION TEST", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"HELMETS: {helmet_count}", 
               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
               (0, 165, 255) if helmet_count > 0 else (0, 0, 255), 2)
    cv2.putText(frame, f"VESTS: {vest_count}", 
               (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
               (0, 255, 0) if vest_count > 0 else (0, 0, 255), 2)
    
    cv2.imshow('PPE Detection - Custom Model Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"ppe_test_{helmet_count}H_{vest_count}V.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Test finished!")