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

# classes=['walk', 'wave', 'stand', 'push up', 'sit']
classes=['', '', 'Deeshee', 'ZVVN', 'duussan', '', 'zogson']
# classes=['2_Degee', 'BARUUN', 'Deeshee', 'ZVVN', 'dooshoo', 'duussan', 'zogson']


# 98.53
try:
    model=load_model('./modelH5/sict.h5')
except :
    print('' , end = '')
# 92.53
# model=load_model('./model/AR_Skeleton_based_V12.h5')


def dataProcess(filename,name):
    outPath=f'./testdata/out/{datetime.today().strftime("%Y%m%d")}'
    imgPath=f'./testdata/img/{datetime.today().strftime("%Y%m%d")}'
    imgPathPose=f'./testdata/imgPose/{datetime.today().strftime("%Y%m%d")}'
    # nowDate=datetime.now()
    nowDate=datetime.now().strftime("%T%m%S")
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
    cap = cv2.VideoCapture(f'/static/uploads/{filename}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{outPath}/{name} {nowDate}.mp4', fourcc, 25.0, shp)
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
            out = cv2.VideoWriter(f'{outPath}/{name}{nowDate}_({cutCounter}).mp4', fourcc, 25.0, shp)
            cv2.imwrite(f'{imgPath}/{name}{nowDate}_({cutCounter}).jpg',frame)
            cv2.imwrite(f'{imgPathPose}/{name}{nowDate}_{cutCounter})(1).jpg',blank_image)
        
            cutCounter+=1
            endPoint+=i
            i=0            
        elif i==25:
            cv2.imwrite(f'{imgPathPose}/{name}{nowDate}_({cutCounter})(0).jpg',blank_image)
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



def getMax(arr):
    max=[0,0]
    
    for i in range(len(arr)):
        if arr[i] > max[0]:
            max[0]=arr[i]
            max[1]=i
    
    return max

def process(filename):
   
    outPath=f'./model/actRec/out/{datetime.today().strftime("%Y%m%d")}'
    imgPath=f'./model/actRec/img/{datetime.today().strftime("%Y%m%d")}'

    nowDate=datetime.now().strftime("%T%m%S")
    try:
        os.mkdir(outPath)
    except :
        print(f'{outPath} already created')
    try:
        os.mkdir(imgPath)
    except :
        print(f'{imgPath} already created')

    h=280
    w=432
    shp=(w,h)
    cap = cv2.VideoCapture(f'./static/uploads/{filename}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outVideoPath=f'{outPath}/{nowDate}.mp4'
    outImgPath=f'{imgPath}/{nowDate}.jpg'
    out = cv2.VideoWriter(outVideoPath, fourcc, 25.0, shp)
    i=0
    endPoint=0
    tic = time.perf_counter()   
    saveImg=False
    Action='...'
    Accuracy=0


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
        img=cv2.resize(blank_image,(224, 224))
        img = img_to_array(img)
        img = img.reshape(1, 224, 224, 3)
        img = img.astype('float32')
        result = model.predict(img)
        max=getMax(result[0])
        Action = classes[max[1]]
        Accuracy = max[0]

        # if 20<=i:
        #     img=cv2.resize(blank_image,(224, 224))
        #     img = img_to_array(img)
        #     img = img.reshape(1, 224, 224, 3)
        #     img = img.astype('float32')
        #     result = model.predict(img)
        #     max=getMax(result[0])
        #     if  max[0]>0.75:  
        #         Action = classes[max[1]]
        #         Accuracy = max[0]
        #     else :
        #         Action = '...'
        #         Accuracy = max[0]
        #     i=0
        # i+=1
        endPoint+=1

        blank_image = cv2.resize(blank_image,shp)
        x=blank_image.shape[0]*0.32
        y=blank_image.shape[1]*0.05
        cv2.putText(blank_image, f'{Action} accuracy = ({round(Accuracy * 100, 2)})', (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 255, 0), 0, cv2.LINE_AA)
        out.write(blank_image)        
        if(endPoint>=125):
            break
        if not saveImg:
            cv2.imwrite(outImgPath,blank_image)
            saveImg = True
        if cv2.waitKey(1) == ord('q'):
            break
    toc = time.perf_counter()
    print(f"timer : \n\t\t{toc - tic:0.4f} seconds")
    out.release()
    cap.release()
    cv2.destroyAllWindows() 
    return [outVideoPath,outImgPath]

def dataProcessRGB(filename,name):
    outPath=f'./testdata/out/{datetime.today().strftime("%Y%m%d")}'
    imgPath=f'./testdata/img/{datetime.today().strftime("%Y%m%d")}'
    imgPathPose=f'./testdata/imgPose/{datetime.today().strftime("%Y%m%d")}'
    # nowDate=datetime.now()
    nowDate=datetime.now().strftime("%T%m%S")
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
    cap = cv2.VideoCapture(f'{filename}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{outPath}/{name} {nowDate}.mp4', fourcc, 25.0, shp)
    i=0
    endPoint=0
    cutCounter=0
    tic = time.perf_counter()
    while cap.isOpened():
        _, frame = cap.read()
        try:
            blank_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        # results = pose.process(RGB)
        
        # blank_image = np.zeros(RGB.shape, np.uint8)

        # mp_drawing.draw_landmarks(
        # blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        # landmark_drawing_spec=draw,
        # connection_drawing_spec=drawline
        # )

        i+=1
        if i==25*2:
            out.release()
            out = cv2.VideoWriter(f'{outPath}/{name}{nowDate}_({cutCounter}).mp4', fourcc, 25.0, shp)
            cv2.imwrite(f'{imgPath}/{name}{nowDate}_({cutCounter}).jpg',frame)
            cv2.imwrite(f'{imgPathPose}/{name}{nowDate}_{cutCounter})(1).jpg',blank_image)
        
            cutCounter+=1
            endPoint+=i
            i=0            
        elif i==25:
            cv2.imwrite(f'{imgPathPose}/{name}{nowDate}_({cutCounter})(0).jpg',blank_image)
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
# for i in os.listdir("./data"):
#     for l in os.listdir(f'./data/{i}'):
#         print(f'./data/{i}/{l}',l.split('.')[0])
#         dataProcessRGB(f'./data/{i}/{l}',l.split('.')[0])
