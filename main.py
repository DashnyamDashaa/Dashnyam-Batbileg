import cv2
from datetime import datetime
from Ds5sD import PoseBased
import sys
def app():
    cap=cv2.VideoCapture("./test videos/b1.webm")
    t=True
    pd=PoseBased()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tf=True
    i,save=0,0
    while True:
        ret,frame = cap.read()
        if not ret:  
            continue 
        orig=frame
        orig=cv2.resize(frame,(frame.shape[1],frame.shape[0]))
        res=pd.process(frame)
        print(res[2])
        frame2 = cv2.vconcat([res[1], res[0]])
        frame3 = cv2.hconcat([orig, cv2.resize(frame2,(frame.shape[1],frame.shape[0]))])
        
        i+=1
        if tf:
            outVideoPath=f'./model/realtime/{datetime.today()}.mp4'
            out = cv2.VideoWriter(outVideoPath, fourcc, 25.0, (frame3.shape[1],frame3.shape[0]))
            out.write(frame3)
            tf=False
            save=0
        else:
            save+=1
            out.write(frame3)
            
        if i>=70:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
def dataProcess():
    pd=PoseBased()
    pd.processData()
if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == '-a':
        app()
    elif args[0] == '-d' or args[0] == '-data':
        dataProcess()