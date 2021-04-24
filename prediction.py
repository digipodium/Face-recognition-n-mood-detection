import cv2
from fer import FER
from fer import Video
import fer
import os

def detect_face(img_path,cascade_path='models/face_finder.xml'):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    # Read the input image
    img = cv2.imread(img_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    return img

def detect_emotion(img_path):
    try:
        detector = FER(mtcnn=True)
        im = cv2.imread(img_path)
        result =  detector.detect_emotions(im)
        return result
    except Exception as e:
        return f'error->{e}'

def detect_emotion_in_video(video_file_path,max_results,freq):
    video = Video(video_file_path,outdir='video_results')
    detector = FER(mtcnn=True)
    return video.analyze(detector,display=False,save_video=True,max_results=max_results,frequency=freq,save_frames=False,zip_images=False)

def detect_face_in_video(video_file_path,cascade_path='models/face_finder.xml'):
    cap = cv2.VideoCapture(video_file_path) 
    name,ext = os.path.splitext(os.path.basename(video_file_path))
    height, width = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),)
    fps = cap.get(cv2.CAP_PROP_FPS)
    outfile = f'{name}_inter{ext}'
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    videowriter = cv2.VideoWriter(
            outfile, fourcc, fps, (width, height), True
        )
    if os.path.exists(cascade_path):
        print('cascade model found')
        faceModel =cv2.CascadeClassifier(cascade_path)
        while True:
            ret, frame =cap.read()
            if not ret:
                break
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
                cv2.putText(frame,"press esc to close",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA,)
            except :
                pass
            # cv2.imshow('face dectection window',frame)
            videowriter.write(frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    return outfile

