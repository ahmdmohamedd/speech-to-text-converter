import whisper
import numpy as np
import sounddevice as sd
import queue
import torch
import time

# Load the pre-trained Whisper model (use a smaller model like 'base' or 'tiny' for faster transcription)
model = whisper.load_model("base")

# A queue to hold the audio data
audio_queue = queue.Queue()

# Audio sampling rate (Whisper models expect 16 kHz input)
SAMPLE_RATE = 16000
BLOCK_SIZE = 16000  # Number of samples per block (1 second of audio)

# Function to continuously collect microphone audio in the background
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Convert the audio input into a 1D array and put it in the queue
    audio_queue.put(indata.copy())

# Start the audio input stream from the microphone
def start_audio_stream():
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE)
    stream.start()
    return stream

# Transcribe live audio from the queue using Whisper
def transcribe_live_audio():
    print("Listening... Speak into the microphone.")

    # Start the audio stream
    stream = start_audio_stream()

    try:
        while True:
            # Wait for a block of audio from the queue
            audio_block = audio_queue.get()

            # Normalize audio to the range expected by Whisper
            audio_block = np.squeeze(audio_block)
            audio_block = audio_block / np.max(np.abs(audio_block)) * 0.9

            # Whisper expects 16 kHz mono audio, so we process chunks of audio block by block
            result = model.transcribe(audio_block, language="en")

            # Print the transcribed text
            print(f"Transcription: {result['text']}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop the audio stream when the script is interrupted
        stream.stop()

# Run the real-time transcription
if __name__ == "__main__":
    transcribe_live_audio()
