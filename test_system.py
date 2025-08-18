#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Emotion Detection - System Test Script
This script helps identify and fix common setup issues
"""

import sys
import os

def test_dependencies():
    """Test all required dependencies"""
    print("="*50)
    print("TESTING DEPENDENCIES")
    print("="*50)

    required_packages = {
        'sounddevice': 'For audio recording',
        'scipy': 'For audio file operations', 
        'librosa': 'For audio feature extraction',
        'numpy': 'For numerical operations',
        'pandas': 'For data manipulation', 
        'sklearn': 'For machine learning (scikit-learn)',
        'joblib': 'For model persistence',
        'matplotlib': 'For plotting',
    }

    missing_packages = []

    for package, purpose in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                package_name = 'scikit-learn'
            else:
                __import__(package)
                package_name = package
            print(f"✓ {package_name} - {purpose}")
        except ImportError:
            print(f"✗ {package_name} - {purpose} - MISSING")
            if package == 'sklearn':
                missing_packages.append('scikit-learn')
            else:
                missing_packages.append(package)

    return missing_packages

def test_file_structure():
    """Test if required files exist"""
    print("\n" + "="*50)
    print("TESTING FILE STRUCTURE")
    print("="*50)

    required_files = {
        'emotion_features.csv': 'Training data',
        'app.py': 'Main application',
        'sort_ravdess.py': 'RAVDESS sorting utility'
    }

    required_dirs = {
        'data': 'Audio data directory',
        'models': 'Model storage directory'
    }

    missing_files = []

    # Check files
    for filename, purpose in required_files.items():
        if os.path.exists(filename):
            print(f"✓ {filename} - {purpose}")
        else:
            print(f"✗ {filename} - {purpose} - MISSING")
            missing_files.append(filename)

    # Check directories
    for dirname, purpose in required_dirs.items():
        if os.path.exists(dirname):
            print(f"✓ {dirname}/ - {purpose}")
        else:
            print(f"✗ {dirname}/ - {purpose} - MISSING")
            os.makedirs(dirname, exist_ok=True)
            print(f"  → Created {dirname}/ directory")

    return missing_files

def test_audio_system():
    """Test audio recording capability"""
    print("\n" + "="*50)
    print("TESTING AUDIO SYSTEM")
    print("="*50)

    try:
        import sounddevice as sd
        print("✓ sounddevice imported successfully")

        # Test audio devices
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]

        if input_devices:
            print(f"✓ Found {len(input_devices)} audio input device(s)")
            print("Available input devices:")
            for i, device in enumerate(input_devices):
                print(f"  {i}: {device['name']}")
        else:
            print("✗ No audio input devices found")
            return False

        # Test basic recording (very short)
        print("Testing basic audio recording...")
        test_recording = sd.rec(int(0.1 * 44100), samplerate=44100, channels=1)
        sd.wait()
        print("✓ Audio recording test successful")

        return True

    except Exception as e:
        print(f"✗ Audio system test failed: {e}")
        return False

def test_model_training():
    """Test if model training works with existing data"""
    print("\n" + "="*50)
    print("TESTING MODEL TRAINING")
    print("="*50)

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # Check if training data exists
        if not os.path.exists('emotion_features.csv'):
            print("✗ emotion_features.csv not found")
            return False

        # Load data
        df = pd.read_csv('emotion_features.csv')
        print(f"✓ Loaded {len(df)} samples from emotion_features.csv")

        # Check data structure
        expected_columns = ['mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 
                           'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9',
                           'mfcc_10', 'mfcc_11', 'mfcc_12', 'pitch', 'energy', 'label']

        if list(df.columns) == expected_columns:
            print("✓ Data structure is correct")
        else:
            print("✗ Data structure mismatch")
            print(f"Expected: {expected_columns}")
            print(f"Found: {list(df.columns)}")
            return False

        # Test model training
        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)  # Smaller for quick test
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✓ Model training successful - Accuracy: {accuracy:.3f}")

        # Test saving model
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/test_model.joblib')
        print("✓ Model saved successfully")

        return True

    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Voice Emotion Detection - System Test")
    print("This script will test your setup and identify issues")
    print()

    # Test 1: Dependencies
    missing_deps = test_dependencies()

    # Test 2: File structure  
    missing_files = test_file_structure()

    # Test 3: Audio system
    audio_works = test_audio_system()

    # Test 4: Model training
    model_works = test_model_training()

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    if missing_deps:
        print("❌ MISSING DEPENDENCIES:")
        print("Install them using: pip install " + " ".join(missing_deps))
        print()

    if missing_files:
        print("❌ MISSING FILES:")
        for f in missing_files:
            print(f"  - {f}")
        print()

    if not audio_works:
        print("❌ AUDIO SYSTEM ISSUES:")
        print("Check microphone permissions and audio drivers")
        print()

    if not model_works:
        print("❌ MODEL TRAINING ISSUES:")
        print("Check data files and dependencies")
        print()

    if not missing_deps and not missing_files and audio_works and model_works:
        print("✅ ALL TESTS PASSED!")
        print("Your system should be ready to run the emotion detection app")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before running the main application")

if __name__ == "__main__":
    main()
