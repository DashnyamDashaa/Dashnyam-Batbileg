import cv2
import mediapipe as mp
import time
import os
from datetime import datetime
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
draw=mp_drawing.DrawingSpec((12, 24, 224), 2, 10)
drawline=mp_drawing.DrawingSpec((224, 224, 224),25, 35)
classes=['2_Degee', 'BARUUN', 'Deeshee', 'ZVVN', 'dooshoo', 'duussan', 'zogson']
model=load_model('./modelH5/AR_Skeleton_based_signal_RGB_V1.h5')

class PoseBased:
    def getMax(self, arr):
        max=[0,0]
        
        for i in range(len(arr)):
            if arr[i] > max[0]:
                max[0]=arr[i]
                max[1]=i
        
        return max
    def process(self,image):
        #input image 
        #out put [ProcessedIMG,[String action, int Acc]]
        try:
            os.mkdir('./actRec')
        except :
            print('' , end = '')
            
        try:
            os.mkdir('./actRec/out')
        except :
            print('' , end = '')

        outPath=f'./actRec/out/{datetime.today().strftime("%Y%m%d")}'

        nowDate=datetime.now().strftime("%T%m%S")
        try:
            os.mkdir(outPath)
        except :
            print(f'' , end = '')

        tic = time.perf_counter()   
        Action='...'
        Accuracy=0
        blank_image = np.zeros(image.shape, np.uint8)
        try:
            RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            return blank_image,["....",0]
        
        results = pose.process(RGB)
        
        mp_drawing.draw_landmarks(
        blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=draw,
        connection_drawing_spec=drawline
        )
        
        mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=draw,
        connection_drawing_spec=drawline
        )
        img=cv2.resize(image,(224, 224))
        img = img_to_array(img)
        img = img.reshape(1, 224, 224, 3)
        img = img.astype('float32')
        result = model.predict(img)
        max=self.getMax(result[0])
        Action = classes[max[1]]
        Accuracy = max[0]
        x=blank_image.shape[0]*0.32
        y=blank_image.shape[1]*0.05
        Ac=""
        if (round(Accuracy * 100, 2)>=55):
            Ac=str(round(Accuracy * 100, 2))
        else:
            Action="Action"
            Ac="???"
        
        cv2.putText(blank_image, f'{Action} accuracy = ({Ac})', (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 255, 0), 0, cv2.LINE_AA)
        cv2.imwrite(f'{outPath}/{nowDate}.jpg',blank_image)
        toc = time.perf_counter()

        return [blank_image,image,[Action,Ac]]