from threading import Thread
import time
import numpy as np
import cv2
from dotenv import load_dotenv
import os
import keras_ocr
from google.auth.credentials import Credentials
from google.cloud import texttospeech
import threading
import os.path
from playsound import playsound
    
load_dotenv()  


class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                #print(self.status)

    def grab_frame(self):
        if self.status:
            return self.frame
        return None  
    def release(self):
        self.capture.release()


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def do_predictions(pipeline, shrunk):
    color = (255, 0, 0)
    thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    prediction_groups = pipeline.recognize([shrunk])
    
    groups = prediction_groups[0]
    text = None
    for group in groups:

        point = np.int32( group[1][0])
        text = group[0]
        cv2.putText(shrunk, text, (point[0], point[1] - 30), font, fontScale, color, thickness)
        rect = np.int32([group[1]])
        #print(rect)
        cv2.polylines(shrunk, rect, isClosed=True, color=color, thickness=thickness)
    return text

def textToSpeech(text):
    path = f'./cached/{text}.mp3'
    if (not os.path.isfile(path)): 
        client = texttospeech.TextToSpeechClient.from_service_account_json('key.json')
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(path, "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "{path}"')
    else:
        print (f'using cache for {text}')
    playsound('./you_spelled.mp3')
    playsound(path)
    


capture_url = os.environ.get("STREAM_URL")
print(capture_url)
cam = ThreadedCamera(capture_url)
#time.sleep(1)
pipeline = keras_ocr.pipeline.Pipeline(max_size=1000)
interval = 5
counter = 0
start_time = time.time()
try:
    while True:
        
        frame = cam.grab_frame()
        shrunk = ResizeWithAspectRatio(frame, 1000)
        text = do_predictions(pipeline, shrunk)
        cv2.imshow('frame', shrunk)
        counter+=1
        if (time.time() - start_time) > interval :
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
            if (text is not None):
                print(text)
                th = threading.Thread(target=textToSpeech, args=(text,))
                th.start()
        if cv2.waitKey(1) == ord('q'):
            break
except Exception as exc:
    print(f'Exception: {exc}')
cam.release()
cv2.destroyAllWindows()