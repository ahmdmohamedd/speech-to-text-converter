# Speech-to-Text Converter with DeepSpeech and Whisper

This repository contains two implementations of a speech-to-text converter using **DeepSpeech** and **Whisper**. Both approaches leverage deep learning models to transcribe audio input into text. The programs can process pre-recorded audio files and capture live audio through a microphone for real-time transcription.

## Table of Contents

- [Overview](#overview)
- [DeepSpeech Setup and Usage](#deepspeech-setup-and-usage)
- [Whisper Setup and Usage](#whisper-setup-and-usage)
- [Requirements](#requirements)
- [Installation](#installation)
- [Contributing](#contributing)

---

## Overview

- **DeepSpeech**: Uses Mozilla's DeepSpeech model, a powerful speech recognition engine built on a neural network architecture. It supports offline transcription from both audio files and live microphone input.
  
- **Whisper**: Developed by OpenAI, Whisper is a state-of-the-art automatic speech recognition (ASR) system that handles multiple languages and noisy environments, offering pre-trained models that are easy to use for both real-time transcription and audio file processing.

### Key Features:
- **DeepSpeech**:
  - Offline transcription from audio files or live audio input.
  - Requires fewer resources than Whisper for real-time transcription.
  
- **Whisper**:
  - Multilingual support.
  - Higher accuracy and better handling of noisy environments.
  - Multiple pre-trained model sizes for different performance needs (speed vs. accuracy).

---

## DeepSpeech Setup and Usage

### Installation

1. First, clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/speech-to-text-converter.git
   cd speech-to-text-converter
   ```

2. Install the required dependencies for DeepSpeech:
   ```bash
   pip install deepspeech numpy scipy sounddevice
   ```

3. Download the pre-trained DeepSpeech model from [DeepSpeech Releases](https://github.com/mozilla/DeepSpeech/releases).

### Running the DeepSpeech Program

To transcribe audio files with DeepSpeech:
   ```bash
   python deepspeech_converter.py --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio your_audio_file.wav
   ```

---

## Whisper Setup and Usage

### Installation

1. Install the necessary dependencies for Whisper:
   ```bash
   pip install git+https://github.com/openai/whisper.git torch numpy sounddevice scipy ffmpeg-python
   ```

### Running the Whisper Program

1. To transcribe audio files with Whisper:
   ```bash
   python whisper_transcribe.py your_audio_file.mp3
   ```

2. For live microphone input transcription:
   ```bash
   python whisper_converter.py
   ```

By default, the program will use the **base** model. You can change the model size (`tiny`, `base`, `small`, `medium`, `large`) by modifying the script.

---

## Requirements

- Python 3.6+ (for DeepSpeech)
- Python 3.7+ (for Whisper)
- **DeepSpeech Model**: You will need to download the pre-trained models from the [DeepSpeech Releases](https://github.com/mozilla/DeepSpeech/releases) page.
- **Whisper Model**: Pre-trained models are included when you install Whisper.

### Additional Libraries:
- `numpy`
- `scipy`
- `sounddevice`
- `torch`
- `ffmpeg-python`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/speech-to-text-converter.git
   cd speech-to-text-converter
   ```

2. Follow the specific setup instructions for each method (DeepSpeech or Whisper).

---

## Contributing

Contributions are welcome! If you have improvements or new features you'd like to add, feel free to open a pull request or issue on GitHub. Make sure to follow the coding standards and provide detailed documentation for any changes.

---

### Author

Developed by [Ahmed Mohamed Kamal Ali](https://www.linkedin.com/in/ahmed-mohamed-kamal-ali).

---

## Notes

- The performance of both DeepSpeech and Whisper depends on the hardware (CPU/GPU) and the size of the models used.
- Whisper generally provides better accuracy for real-time transcription but requires more resources.
- DeepSpeech offers offline transcription and is lighter on resources, making it suitable for low-power devices.
