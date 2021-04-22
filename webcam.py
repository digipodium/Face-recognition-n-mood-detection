import numpy as np
import cv2
import os


def record(camera=0,outfile = 'videos/recording.mp4'):
    cap = cv2.VideoCapture(camera)
    height, width = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(outfile,fourcc, fps, (width,height),True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
            cv2.putText(frame,"press esc to stop recording",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA,)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return None
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return outfile

def detect_face_in_webcam(camera=0,cascade_path='models/face_finder.xml'):

    cap = cv2.VideoCapture(camera) 
    if os.path.exists(cascade_path):
        print('cascade model found')
        faceModel =cv2.CascadeClassifier(cascade_path)
        while True:
            ret, frame =cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                faces =faceModel.detectMultiScale(
                    gray,                               # the b/w image
                    scaleFactor = 1.1,                  # adjust the face near and far from camera 
                    minNeighbors = 5,                   # for the algo to find face relative to other items
                    minSize=(30,30),                    # min box size
                    flags=cv2.CASCADE_SCALE_IMAGE        # dont care
                )

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,h+y),(0,255,0),5,)
                cv2.putText(frame,"press Esc to close",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA,)
            except :
                pass
            cv2.imshow('face dectection window',frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        return False
    return True