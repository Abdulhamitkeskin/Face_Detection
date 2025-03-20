import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)

mpfaceDet=mp.solutions.face_detection
face_det=mpfaceDet.FaceDetection(0.20)

mpDraw = mp.solutions.drawing_utils

while True:
    
    succes,img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results = face_det.process(imgRGB)
    
    if results.detections:
        
        for id,detection in enumerate(results.detections):
            
            bboxC =  detection.location_data.relative_bounding_box
            #print(bboxC)
            h,w,_ = img.shape
            bbox = int(bboxC.xmin*w),int(bboxC.ymin * h),int(bboxC.width*w),int(bboxC.height*h)
            cv2.rectangle(img, bbox,(0,255,255),2)
    
    cv2.imshow("img", img)
    cv2.waitKey(10)