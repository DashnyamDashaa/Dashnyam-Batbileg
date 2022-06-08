from ast import arg
import cv2
from datetime import datetime
from Ds5sD import PoseBased
from Train import acRec
import sys
def app(modelname):
    cap=cv2.VideoCapture(0)
    t=True
    pd=PoseBased(modelname)
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
    pd=PoseBased("DP...")
    pd.processData()
if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == '-a':
        if(len(args)==2):
            train=acRec(args[1])
            if(args[1]=="vgg16"):
                app(args[1])
            elif(args[1]=="sict"):
                app(args[1])
            elif(args[1]=="simple"):
                app(args[1])
            else:
                print("not found "+args[1]+" just for vgg16, sict, simple")
        else:
            print("2 utga baina (-a SaveModelName)")
    elif args[0] == '-d' or args[0] == '-data':
        dataProcess()
    elif args[0] == '-t' or args[0] == '-train':
        if(len(args)==2):
            train=acRec(args[1])
            if(args[1]=="vgg16"):
                train.processVGG16("testdata/dataset/trained")
            elif(args[1]=="sict"):
                train.processsict("testdata/dataset/trained")
            elif(args[1]=="simple"):
                train.processsimple("testdata/dataset/trained")
            else:
                print("not found "+args[1]+" just for vgg16, sict, simple")
        else:
            print("2 utga baina (-train SaveModelName)")