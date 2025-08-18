# Debug script to check your trained model's expected input shape
# Run this first to understand what dimensions your model expects

import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

# Constants matching your original app
SAMPLE_RATE = 44100
DURATION = 6
MODELS_PATH = "models"

def check_model_info():
    """Check what your trained model expects"""
    try:
        model_path = os.path.join(MODELS_PATH, "emotion_cnn_model.h5")
        le_path = os.path.join(MODELS_PATH, "label_encoder.joblib")
        
        if not os.path.exists(model_path):
            print("❌ Model file not found. Please train your model first.")
            return
            
        # Load model
        print("📥 Loading model...")
        model = load_model(model_path)
        le = joblib.load(le_path)
        
        print("✅ Model loaded successfully!")
        print(f"🏷️  Emotions: {list(le.classes_)}")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📈 Model output shape: {model.output_shape}")
        
        # Test with a sample spectrogram
        print("\n🔬 Testing spectrogram extraction...")
        
        # Create a dummy audio signal
        dummy_audio = np.random.randn(SAMPLE_RATE * DURATION)
        
        # Extract spectrogram the same way as in your original code
        spectrogram = librosa.feature.melspectrogram(
            y=dummy_audio, 
            sr=SAMPLE_RATE, 
            n_mels=128
        )
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        print(f"📏 Raw spectrogram shape: {spectrogram_db.shape}")
        
        # Prepare for model input
        spectrogram_input = spectrogram_db[np.newaxis, ..., np.newaxis]
        print(f"🔄 Model input shape (with batch & channel dims): {spectrogram_input.shape}")
        
        expected_shape = model.input_shape
        print(f"🎯 Expected model input shape: {expected_shape}")
        
        # Check if shapes match
        if spectrogram_input.shape[1:] == expected_shape[1:]:
            print("✅ Shapes match! The model should work correctly.")
            
            # Test prediction
            print("\n🤖 Testing model prediction...")
            pred = model.predict(spectrogram_input, verbose=0)
            print(f"✅ Prediction successful! Output shape: {pred.shape}")
            print(f"🎭 Sample probabilities: {pred[0]}")
            
        else:
            print("❌ Shape mismatch detected!")
            print(f"   Got: {spectrogram_input.shape[1:]}")
            print(f"   Expected: {expected_shape[1:]}")
            
            print("\n🔧 Suggested fixes:")
            print("1. Check if your training data was processed with different parameters")
            print("2. Verify the librosa.feature.melspectrogram parameters")
            print("3. Make sure DURATION and SAMPLE_RATE match training")
            
        # Show model summary
        print(f"\n📋 Model Summary:")
        model.summary()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Voice Emotion Detection Model Debugger")
    print("=" * 50)
    check_model_info()