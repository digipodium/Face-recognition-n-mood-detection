IMAGE_FOLDER = 'images'
DATA_FOLDER ='datasets'
DB_PATH ='sqlite:///database/db.sqlite3'
MODEL_FOLDER ='models'
VIDEO_FOLDER ='videos'
VID_RESULT_FOLDER ='video_results'
IMG_RESULT_FOLDER ='image_results'
WEBCAM = 0

FACE_MODEL = f'{MODEL_FOLDER}/face_finder.xml'
VGG_MODEL_WEIGHT = f'{MODEL_FOLDER}/model_weights.xml'

TITLE = "Face recognizer & mood detection"
MENU = ['about project','upload image','upload video','use webcam',]
SUB_MENU = ['detect face', 'detect face and mood']