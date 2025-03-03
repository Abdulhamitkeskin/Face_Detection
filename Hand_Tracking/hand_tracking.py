import cv2
import time
import mediapipe as mp

# Web kamerasını başlat
cap = cv2.VideoCapture(0)

# MediaPipe'in el algılama çözümünü başlat
mpHand = mp.solutions.hands

# El algılama modelini oluştur
hands = mpHand.Hands()
# Modelin bazı ayarları:
# - static_image_mode: Sabit bir görüntü işlemek için kullanılır (bu kodda dinamik olarak ayarlanmış).
# - max_num_hands: Algılanacak maksimum el sayısı (varsayılan 2).
# - min_detection_confidence: Elin algılanması için gereken minimum güven skoru.
# - min_tracking_confidence: Elin izlenebilirliği için gereken minimum güven skoru.

# MediaPipe çizim araçlarını başlat
mpDraw = mp.solutions.drawing_utils

# FPS hesaplaması için başlangıç zamanı
pTime = 0
cTime = 0

while True:
    # Web kamerasından bir kare oku
    success, img = cap.read()
    
    # Görüntüyü RGB formatına çevir (MediaPipe RGB formatında çalışır)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # El algılama modelini görüntü üzerinde çalıştır
    results = hands.process(imgRGB)
    
    # El işaretlerini yazdır (debugging için)
    print(results.multi_hand_landmarks)
    
    # Eğer eller algılandıysa
    if results.multi_hand_landmarks:
        
        # Her bir el için işaretleri çiz
        for handLms in results.multi_hand_landmarks:
            
            # El işaretlerini çiz (kısayol: 'HAND_CONNECTIONS' elin çeşitli noktalarını bağlar)
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            
            # Her bir el işareti için
            for id, lm in enumerate(handLms.landmark):
                # Görüntünün boyutlarını al
                h, w, c = img.shape
                
                # El işareti koordinatlarını piksel değerlerine dönüştür
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Belirli bir işaret (id == 4) için (bu örnekte baş parmak uç noktası)
                if id == 4:
                    # İşaret noktasını kırmızı renkte daire ile işaretle
                    cv2.circle(img, (cx, cy), 9, (0, 0, 255), cv2.FILLED)
    
    # Zamanı güncelle ve FPS hesapla
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # FPS değerini görüntü üzerine yazdır
    cv2.putText(img, "FPS " + str(int(fps)), (10, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
    
    # Görüntüyü ekranda göster
    cv2.imshow("img", img)
    
    # Bir tuşa basana kadar bekle (1 ms)
    cv2.waitKey(1)
