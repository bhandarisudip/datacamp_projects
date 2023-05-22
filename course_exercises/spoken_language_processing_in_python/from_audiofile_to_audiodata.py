# Import the speech recognition library
import speech_recognition as sr

# Instantiate the Recognizer class
recognizer = sr.Recognizer()

# Load up an AudioFile
speech_audio_file = sr.Audio("audiofile.wav")

# Convert an AudioFile to AudioData
with speech_audio_file as source: 
    speech_audio_data = recognizer.record(source,
                                             duration=5,
                                             offset=5)

# Recognize the AudioData with show_all turned on
recognizer.recognize_google(speech_audio_data)
