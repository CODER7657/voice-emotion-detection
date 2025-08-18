# ==============================================================================
# Voice Emotion Detection Application (CNN Version with Data Augmentation)
# ==============================================================================
#
# Author: Gemini
# Description: A command-line application using a Convolutional Neural Network
#              (CNN) with data augmentation to detect emotions from voice.
#              This is the most powerful version of the model.
#
# How to Run (After Update):
# --------------------------
# 1. Install new, required dependencies:
#    pip install tensorflow
#
# 2. Process the audio data with augmentation (this is a new process):
#    python app.py process_features
#
# 3. Train the new CNN model:
#    python app.py train
#
# 4. Run the real-time prediction:
#    python app.py run
#
# ==============================================================================

import os
import sys
import argparse
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# --- Constants ---
SAMPLE_RATE = 44100
DURATION = 6  # seconds
CHANNELS = 1
DATA_PATH = "data"
MODELS_PATH = "models"
FEATURES_PATH = "features" 
TEMP_WAV_FILE = "temp_recording.wav"
EMOTIONS = ["happy", "sad", "angry"]

# --- 1. Data Augmentation Functions ---
def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

def shift_pitch(data, sr, pitch_factor):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def stretch_time(data, stretch_rate):
    return librosa.effects.time_stretch(y=data, rate=stretch_rate)

# --- 2. Feature Extraction (CNN - Spectrogram) ---
def extract_spectrogram(file_path, augment=False):
    """
    Extracts a Mel Spectrogram from an audio file and optionally augments it.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        if augment:
            if np.random.rand() < 0.5:
                y = add_noise(y)
            if np.random.rand() < 0.5:
                pitch_factor = np.random.uniform(-2, 2)
                y = shift_pitch(y, sr, pitch_factor)
            if np.random.rand() < 0.5:
                stretch_rate = np.random.uniform(0.8, 1.2)
                y = stretch_time(y, stretch_rate)

        # Pad or truncate the audio file to be exactly DURATION seconds
        if len(y) < DURATION * sr:
            y = np.pad(y, (0, DURATION * sr - len(y)), mode='constant')
        else:
            y = y[:DURATION * sr]

        # Generate Mel Spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        return spectrogram_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 3. Data Collection Mode ---
def create_data_mode(emotion):
    """
    Guides the user to record audio samples for a specific emotion.
    """
    if emotion not in EMOTIONS:
        print(f"Error: Emotion '{emotion}' not recognized. Choose from {EMOTIONS}.")
        return

    emotion_path = os.path.join(DATA_PATH, emotion)
    os.makedirs(emotion_path, exist_ok=True)

    print(f"\n--- Recording for emotion: {emotion.upper()} ---")
    num_samples = int(input("How many samples would you like to record? (e.g., 10): "))

    for i in range(num_samples):
        print(f"\nRecording sample {i + 1}/{num_samples}...")
        input("Press Enter to start recording...")
        try:
            print(f"Recording for {DURATION} seconds...")
            voice_recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
            sd.wait()
            filename = os.path.join(emotion_path, f"{emotion}_{int(time.time())}.wav")
            write(filename, SAMPLE_RATE, voice_recording)
            print(f"Recording saved to {filename}")
        except Exception as e:
            print(f"An error occurred during recording: {e}")
            break

# --- 4. Feature Processing Mode (CNN with Augmentation) ---
def process_features_mode():
    """
    Processes all audio files, extracts spectrograms, applies augmentation,
    and saves them as numpy arrays.
    """
    all_spectrograms = []
    all_labels = []
    print("\n--- Processing Audio into Spectrograms with Augmentation ---")
    
    for emotion in EMOTIONS:
        emotion_path = os.path.join(DATA_PATH, emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: Directory not found for emotion '{emotion}'. Skipping.")
            continue
        print(f"Processing files for: {emotion.upper()}")
        for filename in os.listdir(emotion_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(emotion_path, filename)
                
                # Original
                spectrogram = extract_spectrogram(file_path, augment=False)
                if spectrogram is not None:
                    all_spectrograms.append(spectrogram)
                    all_labels.append(emotion)
                
                # Augmented
                spectrogram_aug = extract_spectrogram(file_path, augment=True)
                if spectrogram_aug is not None:
                    all_spectrograms.append(spectrogram_aug)
                    all_labels.append(emotion)

    if not all_spectrograms:
        print("No features were extracted. Did you record any data?")
        return

    # Convert to numpy arrays
    X = np.array(all_spectrograms)
    y = np.array(all_labels)

    # Save the processed data
    os.makedirs(FEATURES_PATH, exist_ok=True)
    np.save(os.path.join(FEATURES_PATH, 'X_data.npy'), X)
    np.save(os.path.join(FEATURES_PATH, 'y_data.npy'), y)
    
    print(f"\nSpectrograms and labels successfully saved to the '{FEATURES_PATH}' directory.")
    print(f"Data shape (X): {X.shape}")

# --- 5. Model Training Mode (CNN) ---
def train_model_mode():
    """
    Trains a Convolutional Neural Network (CNN) on the spectrogram data.
    """
    X_path = os.path.join(FEATURES_PATH, 'X_data.npy')
    y_path = os.path.join(FEATURES_PATH, 'y_data.npy')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Processed data not found. Please run 'process_features' first.")
        return

    print("\n--- Training Emotion Detection Model (CNN) ---")
    X = np.load(X_path)
    y_str = np.load(y_path)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    y_categorical = to_categorical(y)
    
    # Add channel dimension for CNN
    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Build the CNN model with more regularization
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(), 
        Dropout(0.25),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(len(EMOTIONS), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    print("Training the CNN... (This will take some time and may use your GPU)")
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[reduce_lr], verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")

    # Save the model and label encoder
    model.save(os.path.join(MODELS_PATH, "emotion_cnn_model.h5"))
    joblib.dump(le, os.path.join(MODELS_PATH, "label_encoder.joblib"))
    print(f"CNN model and label encoder saved to '{MODELS_PATH}' directory.")

# --- 6. Real-Time Prediction Mode (CLI) ---
def run_cli_mode():
    """
    Runs a loop to record audio, convert to spectrogram, and predict emotion using the CNN.
    """
    model_path = os.path.join(MODELS_PATH, "emotion_cnn_model.h5")
    le_path = os.path.join(MODELS_PATH, "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print(f"Error: Model or Label Encoder not found. Please train the model first.")
        return
    
    model = load_model(model_path)
    le = joblib.load(le_path)
    print("\n--- Real-Time Emotion Detection (CNN) ---")
    
    while True:
        input("Press Enter to start recording, or Ctrl+C to exit...")
        try:
            print(f"Recording for {DURATION} seconds...")
            audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
            sd.wait()
            
            print("Recording finished. Analyzing...")
            write(TEMP_WAV_FILE, SAMPLE_RATE, audio_data)
            
            spectrogram = extract_spectrogram(TEMP_WAV_FILE, augment=False)
            if spectrogram is not None:
                # Reshape for model prediction
                spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
                
                probabilities = model.predict(spectrogram)[0]
                prediction_idx = np.argmax(probabilities)
                prediction = le.inverse_transform([prediction_idx])[0]
                
                print("\n--- Prediction Result ---")
                print(f"Detected Emotion: {prediction.upper()}")
                
                prob_text = "Confidence: ["
                for i, emotion in enumerate(le.classes_):
                    prob_text += f"{emotion.capitalize()}: {probabilities[i]*100:.0f}%, "
                print(prob_text.strip(", ") + "]")
                print("-" * 25 + "\n")

            else:
                print("Could not extract features from the recording.")

        except KeyboardInterrupt:
            print("\nExiting application.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Voice Emotion Detection Tool")
    parser.add_argument('mode', choices=['create_data', 'process_features', 'train', 'run'],
                        help="The mode to run the script in.")
    parser.add_argument('--emotion', choices=EMOTIONS,
                        help="The emotion to record for (used with 'create_data' mode).")

    args = parser.parse_args()

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

    if args.mode == 'create_data':
        if not args.emotion:
            print("Error: --emotion argument is required for 'create_data' mode.")
            sys.exit(1)
        create_data_mode(args.emotion)
    elif args.mode == 'process_features':
        process_features_mode()
    elif args.mode == 'train':
        train_model_mode()
    elif args.mode == 'run':
        run_cli_mode()

if __name__ == "__main__":
    main()
