import numpy as np
import wave
import deepspeech
import scipy.io.wavfile as wav

# Load the pre-trained DeepSpeech model
model_path = 'deepspeech-0.9.3-models.pbmm'
scorer_path = 'deepspeech-0.9.3-models.scorer'
model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)

# Function to convert audio to text
def convert_speech_to_text(audio_file):
    # Read the audio file
    with wave.open(audio_file, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
    
    # Convert the audio buffer into numpy array
    audio = np.frombuffer(buffer, dtype=np.int16)
    
    # Use DeepSpeech model for transcription
    text = model.stt(audio)
    return text

# Test with an audio file (make sure it's a .wav file)
audio_file = 'your_audio_file.wav'
transcription = convert_speech_to_text(audio_file)
print("Transcription: ", transcription)
