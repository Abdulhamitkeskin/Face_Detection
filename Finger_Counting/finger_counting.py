import cv2
import mediapipe as mp

# Web kamerasından görüntü almak için VideoCapture nesnesini oluşturur.
cap = cv2.VideoCapture(0)
# VideoCapture nesnesinin genişlik ve yüksekliğini ayarlar.
cap.set(3, 640)  # Genişlik: 640 piksel
cap.set(4, 480)  # Yükseklik: 480 piksel

# Mediapipe el tanıma modülünü başlatır.
mpHand = mp.solutions.hands
hands = mpHand.Hands()
# El noktalarını çizmek için kullanılır.
mpDraw = mp.solutions.drawing_utils

# Parmak uçlarının ID'lerini tanımlar.
tip_Ids = [4, 8, 12, 16, 20]  # Başparmak, işaret parmağı, orta parmak, yüzük parmağı, serçe parmak

while True:
    # Web kamerasından görüntü alır.
    success, img = cap.read()
    # Görüntü alınamadıysa döngüyü kırar ve hata mesajı verir.
    if not success:
        print("Web kamerasından görüntü alınamadı.")
        break

    # Görüntüyü RGB formatına çevirir, çünkü Mediapipe RGB formatında çalışır.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Mediapipe el tanıma modelini uygular.
    results = hands.process(imgRGB)
    
    # El noktalarını saklamak için boş bir liste oluşturur.
    lmList = []
    if results.multi_hand_landmarks:
        # Eğer el noktaları algılanmışsa:
        for handLms in results.multi_hand_landmarks:
            # El noktalarını ve bağlantılarını çizerek görüntüyü günceller.
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            
            # Her bir el noktası için:
            for id, lm in enumerate(handLms.landmark):
                # Görüntü boyutlarını alır.
                h, w, c = img.shape
                # El noktasının koordinatlarını hesaplar.
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Noktaların ID, x, y koordinatlarını lmList listesine ekler.
                lmList.append([id, cx, cy])
    
    # Eğer lmList listesi boş değilse (yani en az bir el noktası algılandıysa):
    if len(lmList) != 0:
        # Parmakları belirlemek için bir liste oluşturur.
        fingers = []
        
        # Sağ veya sol eli ayırt etmek için başparmak konumunu kontrol eder
        if lmList[tip_Ids[0]][1] < lmList[tip_Ids[1]][1]:
            # Sağ el: Başparmak x koordinatı daha küçükse sağ el
            # Başparmak açık mı kapalı mı kontrol edilir
            if lmList[tip_Ids[0]][1] < lmList[tip_Ids[0] - 1][1]:
                fingers.append(1)  # Başparmak açık
            else:
                fingers.append(0)  # Başparmak kapalı
        else:
            # Sol el: Başparmak x koordinatı büyükse sol el
            if lmList[tip_Ids[0]][1] > lmList[tip_Ids[0] - 1][1]:
                fingers.append(1)  # Başparmak açık
            else:
                fingers.append(0)  # Başparmak kapalı
            
        # Diğer parmaklar için:
        # Her parmak için uç nokta ile alt nokta arasındaki y koordinatını karşılaştırır.
        # Eğer uç nokta yukarıdaysa, parmak açık olarak kabul edilir.
        for id in range(1, 5):
            if lmList[tip_Ids[id]][2] < lmList[tip_Ids[id] - 2][2]:
                fingers.append(1)  # Parmak açık
            else:
                fingers.append(0)  # Parmak kapalı
            
        # Açık parmakları sayar ve bu sayıyı görüntüye yazar.
        totalF = fingers.count(1)
        cv2.putText(img, str(totalF), (30, 125), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 8)
    
    # Sonuç görüntüsünü gösterir.
    cv2.imshow("img", img)
    # 1 ms bekler ve herhangi bir tuşa basılmasını bekler.
    cv2.waitKey(1)
