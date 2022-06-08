import cv2
from Pose import process
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
# classes=['2_Degee', 'BARUUN', 'Deeshee', 'ZVVN', 'dooshoo', 'duussan', 'zogson']
# model=load_model('./modelH5/AR_Skeleton_based_signal_RGB_V1.h5')

class PoseBased:
    def __init__(self,modelName):
        if(modelName!="DP..."):
            self.model=load_model("./modelH5/"+modelName+".h5")
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
        result = self.model.predict(img)
        max=self.getMax(result[0])
        Action = str(max[1])
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
    def data(self,filePath,action):
        print(filePath)
        outPath=f'./testdata/out/{datetime.today().strftime("%Y%m%d")}'
        imgPath=f'./testdata/img/{datetime.today().strftime("%Y%m%d")}'
        imgPathPose=f'./testdata/imgPose/{datetime.today().strftime("%Y%m%d")}'
        # nowDate=datetime.now()
        nowDate=datetime.now().strftime(" %T%m%S")
        try:
            os.mkdir(outPath)
        except :
            print(f'{outPath} already created')
        try:
            os.mkdir(imgPath)
        except :
            print(f'{imgPath} already created')
        try:
            os.mkdir(imgPathPose)
        except :
            print(f'{imgPathPose} already created')
        # create capture object
        h=280
        w=432
        shp=(w,h)
        cap = cv2.VideoCapture(filePath)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{outPath}/{action} {nowDate}.mp4', fourcc, 25.0, shp)
        i=0
        endPoint=0
        cutCounter=0
        tic = time.perf_counter()
        while cap.isOpened():
            _, frame = cap.read()
            try:
                RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                break
            
            results = pose.process(RGB)
            
            blank_image = np.zeros(RGB.shape, np.uint8)

            mp_drawing.draw_landmarks(
            blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=draw,
            connection_drawing_spec=drawline
            )

            i+=1
            if i==25*2:
                out.release()
                out = cv2.VideoWriter(f'{outPath}/{action}{nowDate}_({cutCounter}).mp4', fourcc, 25.0, shp)
                cv2.imwrite(f'{imgPath}/{action}{nowDate}_({cutCounter}).jpg',frame)
                cv2.imwrite(f'{imgPathPose}/{action}{nowDate}_{cutCounter})(1).jpg',blank_image)
            
                cutCounter+=1
                endPoint+=i
                i=0            
            elif i==25:
                cv2.imwrite(f'{imgPathPose}/{action}{nowDate}_({cutCounter})(0).jpg',blank_image)
            if(endPoint>=25*3*60):
                break
            blank_image = cv2.resize(blank_image,shp)
            out.write(blank_image)
        
            if cv2.waitKey(1) == ord('q'):
                break
        toc = time.perf_counter()
        print(f"timer : \n\t\t{toc - tic:0.4f} seconds")
        out.release()
        cap.release()
        cv2.destroyAllWindows()
    def processData(self):
        if(len(os.listdir("./data"))>1):
            for folder in os.listdir("./data"):
                if(folder=="readme.md"):
                    continue
                for video in os.listdir("./data/"+folder):
                    self.data("./data/"+folder+"/"+video,folder)
        else :
            print("Bolowsruulah data hooson baina")

