"""Synthesizes speech from the input string of text or ssml.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.auth.credentials import Credentials
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
import io
from playsound import playsound
playsound("output.mp3")
# Instantiates a client
#Credentials c = Credentials()
client = texttospeech.TextToSpeechClient.from_service_account_json('key.json')

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text="Hello, World!")

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
)


#song = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
#song = AudioSegment.from_file("output.mp3", format="mp3")
#play(song)
# The response's audio_content is binary.
with open("output.mp3", "wb") as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')