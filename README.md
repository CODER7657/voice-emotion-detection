# Voice Emotion Detection System

A real-time voice emotion detection system that uses a Convolutional Neural Network (CNN) to classify emotions (happy, sad, angry) from live audio input. This system provides both terminal-based and GUI-based interfaces for emotion prediction from voice data.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
- [Usage](#usage)
- [Missing Models/Features](#missing-modelsfeatures)
- [Git and Large Files](#git-and-large-files)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## Overview

This project implements a voice emotion detection system with the following capabilities:
- **Real-time emotion classification** from microphone input
- **Visual feedback** with waveform and spectrogram displays
- **CNN-based model** trained on audio features
- **Support for multiple emotions**: happy, sad, angry
- **Flexible interfaces**: command-line and GUI applications

### Key Features
- Live audio processing and emotion prediction
- Pre-trained model for immediate use
- Training scripts for custom datasets
- Visual analysis tools with spectrograms
- Cross-platform compatibility

## Installation

### Prerequisites
- Python 3.7 or higher
- Microphone access for live recording
- At least 2GB of free disk space (for models and datasets)

### Step 1: Clone the Repository
```bash
git clone https://github.com/CODER7657/voice-emotion-detection.git
cd voice-emotion-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python test_system.py
```

## Project Structure

```
voice-emotion-detection/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
│
├── app.py                   # Main terminal application
├── emotion_gui.py           # GUI application with visualizations
├── debug_model.py           # Model debugging utilities
├── test_system.py           # System verification script
├── sort_ravdess.py          # Dataset organization helper
│
├── models/                  # Model files (may be missing initially)
│   ├── emotion_cnn_model.h5 # Trained CNN model
│   └── label_encoder.pkl    # Label encoder for classes
│
├── features/                # Processed feature files
│   └── processed_features.pkl # Extracted audio features
│
├── datasets/                # Training datasets (create if needed)
│   └── RAVDESS/            # RAVDESS emotion dataset
│
└── emotion_features.csv     # Feature extraction results
```

### File Descriptions

**Core Applications:**
- `app.py`: Command-line interface for emotion prediction and model training
- `emotion_gui.py`: Graphical interface with real-time waveform and spectrogram visualization

**Utility Scripts:**
- `debug_model.py`: Tools for model analysis and debugging
- `test_system.py`: Verification script to check if all components work
- `sort_ravdess.py`: Helper to organize RAVDESS dataset files by emotion

**Data Files:**
- `models/`: Contains trained neural network model and label encoders
- `features/`: Preprocessed audio features ready for training/prediction
- `emotion_features.csv`: Tabular feature data for analysis

## Setup Guide

### For Complete Beginners

1. **Install Python**: Download from [python.org](https://python.org) (version 3.7+)
2. **Open Terminal/Command Prompt**:
   - Windows: Press `Win + R`, type `cmd`, press Enter
   - macOS: Press `Cmd + Space`, type "terminal", press Enter
   - Linux: Press `Ctrl + Alt + T`

3. **Navigate to Project Folder**:
   ```bash
   cd path/to/voice-emotion-detection
   ```

4. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Test Installation**:
   ```bash
   python test_system.py
   ```

### Handling Permission Issues

**On macOS/Linux**, you might need:
```bash
sudo pip install -r requirements.txt
```

**For microphone access**, ensure your system allows Python apps to access the microphone.

## Usage

### Quick Start

**Option 1: GUI Application (Recommended for beginners)**
```bash
python emotion_gui.py
```
Features:
- Real-time emotion prediction
- Live waveform display
- Spectrogram visualization
- Easy-to-use interface

**Option 2: Terminal Application**
```bash
python app.py
```
Features:
- Command-line interface
- Faster startup
- Suitable for automation

### Advanced Usage

**Train New Model:**
```bash
python app.py --train --dataset path/to/dataset
```

**Test Model Accuracy:**
```bash
python debug_model.py --evaluate
```

**Organize RAVDESS Dataset:**
```bash
python sort_ravdess.py --input path/to/ravdess --output datasets/RAVDESS
```

### Sample Commands

**Basic Prediction:**
```bash
# Start GUI with live prediction
python emotion_gui.py

# Start terminal prediction
python app.py --predict
```

**Training from Scratch:**
```bash
# Train with default settings
python app.py --train

# Train with custom parameters
python app.py --train --epochs 50 --batch_size 32
```

**System Testing:**
```bash
# Test all components
python test_system.py --full

# Test only model loading
python test_system.py --model-only
```

## Missing Models/Features

### If Models Folder is Empty

The `models/` folder contains large files that may not be included in the repository. Here's how to handle missing models:

#### Option 1: Download Pre-trained Models
1. Check the [Releases page](https://github.com/CODER7657/voice-emotion-detection/releases) for model downloads
2. Download `emotion_cnn_model.h5` and `label_encoder.pkl`
3. Place them in the `models/` folder

#### Option 2: Train from Scratch

**Step 1: Get Dataset**
```bash
# Download RAVDESS dataset (example)
# Visit: https://zenodo.org/record/1188976
# Extract to datasets/RAVDESS/
```

**Step 2: Prepare Data**
```bash
python sort_ravdess.py --input path/to/ravdess --output datasets/RAVDESS
```

**Step 3: Train Model**
```bash
python app.py --train --dataset datasets/RAVDESS
```

**Training Requirements:**
- At least 4GB RAM
- 30+ minutes training time
- Properly organized dataset

### If Features are Missing

If `emotion_features.csv` or `features/` folder is missing:

```bash
# Extract features from dataset
python app.py --extract-features --dataset datasets/RAVDESS

# This will create:
# - emotion_features.csv
# - features/processed_features.pkl
```

### Dataset Structure

For training from scratch, organize your dataset like this:
```
datasets/
└── RAVDESS/
    ├── happy/
    │   ├── audio1.wav
    │   └── audio2.wav
    ├── sad/
    │   ├── audio3.wav
    │   └── audio4.wav
    └── angry/
        ├── audio5.wav
        └── audio6.wav
```

## Git and Large Files

### .gitignore Recommendations

Create or update `.gitignore` with:
```gitignore
# Large model files
models/*.h5
models/*.pkl
features/*.pkl

# Dataset files
datasets/
*.wav
*.mp3
*.flac

# Python
__pycache__/
*.pyc
*.pyo
venv/
.env

# OS files
.DS_Store
Thumbs.db

# Jupyter notebooks
.ipynb_checkpoints/
```

### Handling Large Files

**For Model Files (>100MB):**
1. **Use Git LFS** (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.h5"
   git lfs track "*.pkl"
   git add .gitattributes
   git commit -m "Add LFS tracking"
   ```

2. **Use External Storage**:
   - Upload large files to cloud storage (Google Drive, Dropbox)
   - Include download links in README
   - Use download scripts

3. **Split Large Files**:
   ```python
   # Example: split large model file
   import pickle
   
   # Save model in chunks
   model.save('model_part1.h5')
   # ... additional parts
   ```

**Recommended Workflow:**
```bash
# Clone repo
git clone <repo-url>
cd voice-emotion-detection

# Download large files separately
# (from releases or external links)

# Install and run
pip install -r requirements.txt
python test_system.py
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Solution: Install missing packages
pip install -r requirements.txt

# For specific packages:
pip install tensorflow librosa soundfile
```

**2. Microphone Not Working**
```bash
# Test microphone access
python -c "import sounddevice; print(sounddevice.query_devices())"

# Solutions:
# - Check system microphone permissions
# - Try different audio device
# - Install audio drivers
```

**3. Model Loading Errors**
```bash
# Check if model files exist
ls -la models/

# If missing, train new model:
python app.py --train
```

**4. Memory Errors**
```bash
# Reduce batch size in training
python app.py --train --batch_size 16

# Close other applications
# Use smaller model architecture
```

**5. Audio Format Issues**
```bash
# Install additional codecs
pip install librosa[all]

# Convert audio files
# ffmpeg -i input.mp3 output.wav
```

### Platform-Specific Issues

**Windows:**
- Install Microsoft Visual C++ Redistributable
- Use `python` instead of `python3`
- Check Windows Defender permissions

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Grant microphone permissions in System Preferences
- Use `python3` command

**Linux:**
- Install audio system dependencies:
  ```bash
  sudo apt-get install portaudio19-dev python3-pyaudio
  ```
- Check ALSA/PulseAudio configuration

### Performance Issues

**Slow Prediction:**
- Use smaller model architecture
- Reduce audio sample rate
- Close unnecessary applications

**High CPU Usage:**
- Reduce real-time processing frequency
- Use threading for GUI updates
- Optimize feature extraction

### Debug Mode

```bash
# Run with debug information
python debug_model.py --verbose

# Test individual components
python test_system.py --component audio
python test_system.py --component model
python test_system.py --component gui
```

## Resources

### Datasets
- **RAVDESS**: [Emotional Speech Database](https://zenodo.org/record/1188976)
- **TESS**: [Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487)
- **CREMA-D**: [Multimodal Emotion Dataset](https://github.com/CheyneyComputerScience/CREMA-D)

### Documentation
- **TensorFlow**: [tensorflow.org](https://tensorflow.org)
- **Librosa**: [Audio Analysis Library](https://librosa.org)
- **PyQt5**: [GUI Framework](https://riverbankcomputing.com/software/pyqt/)

### Research Papers
- "Speech Emotion Recognition using CNN" 
- "Deep Learning for Audio Signal Processing"
- "Real-time Emotion Recognition from Speech"

### Tutorials
- [Audio Processing with Python](https://realpython.com/python-audio/)
- [CNN for Audio Classification](https://towardsdatascience.com/audio-classification-with-cnn)
- [Building Audio Applications](https://python-sounddevice.readthedocs.io/)

### Community
- **Issues**: Report bugs on [GitHub Issues](https://github.com/CODER7657/voice-emotion-detection/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/CODER7657/voice-emotion-detection/discussions)
- **Stack Overflow**: Tag questions with `voice-recognition`, `emotion-detection`

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### License
This project is open source. Check the LICENSE file for details.

---

**Need Help?** 
- Check the troubleshooting section above
- Run `python test_system.py` to diagnose issues
- Open an issue on GitHub with error details

**Quick Start Summary:**
1. `git clone <repo>`
2. `cd voice-emotion-detection`
3. `pip install -r requirements.txt`
4. `python emotion_gui.py`
