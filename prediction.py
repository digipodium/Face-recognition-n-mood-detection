import cv2
from fer import FER
from fer import Video
import fer

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

def detect_emotion_in_video(video_file_path):
    video = Video(video_file_path,outdir='video_results')
    detector = FER(mtcnn=True)
    return video.analyze(detector,display=True,save_video=True)
