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
from imutils.video import WebcamVideoStream
import imutils
from fps import FPS


def do_predictions(pipeline, shrunk):
    color = (255, 0, 0)
    thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    prediction_groups = pipeline.recognize([shrunk])
    
    groups = prediction_groups[0]
    
    for group in groups:

        point = np.int32( group[1][0])
        text = group[0]
        cv2.putText(shrunk, text, (point[0], point[1] - 30), font, fontScale, color, thickness)
        rect = np.int32([group[1]])
        #print(rect)
        cv2.polylines(shrunk, rect, isClosed=True, color=color, thickness=thickness)
    if (len(groups) > 0):
        return groups[0][0]
    else:
        return None

def textToSpeech(text):
    path = f'./cached/{text}.mp3'
    if (not os.path.isfile(path)): 
        client = texttospeech.TextToSpeechClient.from_service_account_json('key.json')
        hyphenated = "-".join([char for char in text])
        synthesis_input = texttospeech.SynthesisInput(text=f'{hyphenated} spells "{text}"')
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
            print(f'Fetched Audio content written to file "{path}"')
    # else:
    #     print (f'using cache for {text}')
    playsound(path)
    


capture_url = os.environ.get("STREAM_URL")
print(capture_url)
cam = WebcamVideoStream(capture_url).start()

#time.sleep(1)
pipeline = keras_ocr.pipeline.Pipeline(max_size=1000)
fps = FPS(5).start()
try:
    while True:
        
        frame = cam.read()
        shrunk =  imutils.resize(frame, 1000)
        text = do_predictions(pipeline, shrunk)
        cv2.imshow('frame', shrunk)
        if (fps.updateAndPrintAndReset()):
            if (text is not None):
                #print(text)
                th = threading.Thread(target=textToSpeech, args=(text,))
                th.start()
        if cv2.waitKey(1) == ord('q'):
            break
except Exception as exc:
    print(f'Exception: {exc}')

cam.stop()
cam.stream.release()
cv2.destroyAllWindows()