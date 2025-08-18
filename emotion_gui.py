# ==============================================================================
# Voice Emotion Detection GUI Application
# ==============================================================================
#
# A modern GUI interface for the voice emotion detection model
# Features: Record button, real-time spectrogram, waveform visualization,
# and emotion prediction with confidence levels
#
# Requirements: pip install tkinter matplotlib numpy librosa tensorflow sounddevice scipy
#
# ==============================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sounddevice as sd
from scipy.io.wavfile import write
from scipy import ndimage
import librosa
import os
from tensorflow.keras.models import load_model
import joblib
import time

class VoiceEmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Constants (matching your model)
        self.SAMPLE_RATE = 44100
        self.DURATION = 6
        self.CHANNELS = 1
        self.TEMP_WAV_FILE = "temp_recording.wav"
        self.EMOTIONS = ["happy", "sad", "angry"]
        self.MODELS_PATH = "models"
        
        # State variables
        self.is_recording = False
        self.audio_data = None
        self.model = None
        self.label_encoder = None
        
        # Load model and label encoder
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
        
    def load_model(self):
        """Load the trained CNN model and label encoder"""
        try:
            model_path = os.path.join(self.MODELS_PATH, "emotion_cnn_model.h5")
            le_path = os.path.join(self.MODELS_PATH, "label_encoder.joblib")
            
            if os.path.exists(model_path) and os.path.exists(le_path):
                self.model = load_model(model_path)
                self.label_encoder = joblib.load(le_path)
                print("Model and label encoder loaded successfully!")
                
                # Print model input shape for debugging
                if self.model:
                    input_shape = self.model.input_shape
                    print(f"Model expects input shape: {input_shape}")
                    
            else:
                messagebox.showerror("Error", "Model files not found. Please train the model first.")
                
        except Exception as e:
            print(f"Model loading error: {e}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, 
                              text="🎤 Voice Emotion Detection System", 
                              font=('Arial', 24, 'bold'),
                              fg='#ecf0f1', bg='#2c3e50')
        title_label.pack()
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#34495e', relief='ridge', bd=2)
        control_frame.pack(pady=10, padx=20, fill='x')
        
        # Record button
        self.record_btn = tk.Button(control_frame, 
                                   text="🔴 Start Recording", 
                                   font=('Arial', 14, 'bold'),
                                   bg='#e74c3c', fg='white',
                                   relief='raised', bd=3,
                                   command=self.toggle_recording,
                                   width=20, height=2)
        self.record_btn.pack(side='left', padx=20, pady=10)
        
        # Status label
        self.status_label = tk.Label(control_frame,
                                   text="Ready to record...",
                                   font=('Arial', 12),
                                   fg='#ecf0f1', bg='#34495e')
        self.status_label.pack(side='left', padx=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, 
                                       length=200, 
                                       mode='determinate')
        self.progress.pack(side='right', padx=20, pady=10)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#2c3e50')
        content_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left panel - Visualizations
        left_panel = tk.Frame(content_frame, bg='#34495e', relief='ridge', bd=2)
        left_panel.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        viz_title = tk.Label(left_panel, 
                           text="Audio Visualizations", 
                           font=('Arial', 16, 'bold'),
                           fg='#ecf0f1', bg='#34495e')
        viz_title.pack(pady=10)
        
        # Create matplotlib figure for visualizations
        self.fig = Figure(figsize=(8, 8), facecolor='#34495e')
        self.canvas = FigureCanvasTkAgg(self.fig, left_panel)
        self.canvas.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)
        
        # Waveform subplot
        self.ax_wave = self.fig.add_subplot(2, 1, 1)
        self.ax_wave.set_facecolor('#2c3e50')
        self.ax_wave.set_title('Waveform', color='white', fontsize=12)
        self.ax_wave.set_xlabel('Time (s)', color='white')
        self.ax_wave.set_ylabel('Amplitude', color='white')
        self.ax_wave.tick_params(colors='white')
        
        # Spectrogram subplot
        self.ax_spec = self.fig.add_subplot(2, 1, 2)
        self.ax_spec.set_facecolor('#2c3e50')
        self.ax_spec.set_title('Mel Spectrogram', color='white', fontsize=12)
        self.ax_spec.set_xlabel('Time (s)', color='white')
        self.ax_spec.set_ylabel('Mel Frequency', color='white')
        self.ax_spec.tick_params(colors='white')
        
        self.fig.tight_layout()
        
        # Right panel - Results
        right_panel = tk.Frame(content_frame, bg='#34495e', relief='ridge', bd=2, width=300)
        right_panel.pack(side='right', fill='y', padx=(10, 0))
        right_panel.pack_propagate(False)
        
        results_title = tk.Label(right_panel,
                               text="Emotion Detection Results",
                               font=('Arial', 16, 'bold'),
                               fg='#ecf0f1', bg='#34495e')
        results_title.pack(pady=15)
        
        # Prediction result
        self.prediction_frame = tk.Frame(right_panel, bg='#2c3e50', relief='ridge', bd=2)
        self.prediction_frame.pack(pady=10, padx=15, fill='x')
        
        self.prediction_label = tk.Label(self.prediction_frame,
                                       text="No prediction yet",
                                       font=('Arial', 18, 'bold'),
                                       fg='#ecf0f1', bg='#2c3e50')
        self.prediction_label.pack(pady=20)
        
        # Confidence levels
        confidence_title = tk.Label(right_panel,
                                  text="Confidence Levels",
                                  font=('Arial', 14, 'bold'),
                                  fg='#ecf0f1', bg='#34495e')
        confidence_title.pack(pady=(20, 10))
        
        # Confidence bars for each emotion
        self.confidence_frame = tk.Frame(right_panel, bg='#34495e')
        self.confidence_frame.pack(pady=10, padx=15, fill='x')
        
        self.confidence_vars = {}
        self.confidence_bars = {}
        
        emotion_colors = {'happy': '#f39c12', 'sad': '#3498db', 'angry': '#e74c3c'}
        
        for emotion in self.EMOTIONS:
            emotion_frame = tk.Frame(self.confidence_frame, bg='#34495e')
            emotion_frame.pack(fill='x', pady=5)
            
            emotion_label = tk.Label(emotion_frame,
                                   text=f"{emotion.capitalize()}:",
                                   font=('Arial', 10),
                                   fg='white', bg='#34495e',
                                   width=8, anchor='w')
            emotion_label.pack(side='left')
            
            self.confidence_vars[emotion] = tk.DoubleVar()
            
            confidence_bar = ttk.Progressbar(emotion_frame,
                                           variable=self.confidence_vars[emotion],
                                           maximum=100,
                                           length=150)
            confidence_bar.pack(side='left', padx=5)
            self.confidence_bars[emotion] = confidence_bar
            
            confidence_text = tk.Label(emotion_frame,
                                     textvariable=self.confidence_vars[emotion],
                                     font=('Arial', 9),
                                     fg='white', bg='#34495e',
                                     width=5)
            confidence_text.pack(side='right')
        
        # Initialize empty visualizations
        self.update_visualizations()
        
    def extract_spectrogram(self, file_path):
        """Extract spectrogram from audio file (matching your model's method exactly)"""
        try:
            y, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, duration=self.DURATION)
            
            # Pad or truncate the audio file to be exactly DURATION seconds
            if len(y) < self.DURATION * sr:
                y = np.pad(y, (0, self.DURATION * sr - len(y)), mode='constant')
            else:
                y = y[:self.DURATION * sr]
            
            # Generate Mel Spectrogram with parameters that will give us the right shape
            # Your model expects (128, 345, 1), so we need to calculate the right hop_length
            # For 6 seconds at 44100 Hz = 264600 samples
            # To get 345 time frames: hop_length = 264600 / (345 - 1) ≈ 768
            spectrogram = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=128,
                hop_length=768,  # Adjusted to match your trained model
                n_fft=2048
            )
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            
            print(f"Spectrogram shape: {spectrogram_db.shape}")  # Debug info
            
            # If the shape is still not exactly right, crop or pad to match exactly
            if spectrogram_db.shape[1] != 345:
                if spectrogram_db.shape[1] > 345:
                    # Crop to 345
                    spectrogram_db = spectrogram_db[:, :345]
                else:
                    # Pad to 345
                    pad_width = 345 - spectrogram_db.shape[1]
                    spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
            
            print(f"Final spectrogram shape: {spectrogram_db.shape}")  # Debug info
            
            return spectrogram_db, y
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, None
    
    def toggle_recording(self):
        """Toggle between start and stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording in a separate thread"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please ensure the model is trained.")
            return
            
        self.is_recording = True
        self.record_btn.config(text="⏹️ Stop Recording", bg='#27ae60')
        self.status_label.config(text=f"Recording for {self.DURATION} seconds...")
        self.progress.config(maximum=100, value=0)
        
        # Clear previous results
        self.prediction_label.config(text="Recording...")
        for emotion in self.EMOTIONS:
            self.confidence_vars[emotion].set(0)
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start progress update
        self.update_progress()
    
    def stop_recording(self):
        """Stop recording (handled automatically after duration)"""
        pass
    
    def record_audio(self):
        """Record audio for the specified duration"""
        try:
            self.audio_data = sd.rec(int(self.DURATION * self.SAMPLE_RATE), 
                                   samplerate=self.SAMPLE_RATE, 
                                   channels=self.CHANNELS, 
                                   dtype='float32')
            sd.wait()
            
            # Save the recording
            write(self.TEMP_WAV_FILE, self.SAMPLE_RATE, self.audio_data)
            
            # Process the recording
            self.root.after(0, self.process_recording)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recording failed: {str(e)}"))
            self.root.after(0, self.reset_ui)
    
    def update_progress(self):
        """Update progress bar during recording"""
        if self.is_recording:
            current_value = self.progress['value']
            if current_value < 100:
                self.progress['value'] = current_value + (100 / (self.DURATION * 10))
                self.root.after(100, self.update_progress)
    
    def process_recording(self):
        """Process the recorded audio and make prediction"""
        try:
            self.status_label.config(text="Processing audio...")
            
            # Extract spectrogram and waveform
            spectrogram, waveform = self.extract_spectrogram(self.TEMP_WAV_FILE)
            
            if spectrogram is not None and waveform is not None:
                # Update visualizations
                self.update_visualizations(waveform, spectrogram)
                
                # Prepare input for the model (match your training preprocessing exactly)
                spectrogram_input = spectrogram[np.newaxis, ..., np.newaxis]
                print(f"Model input shape: {spectrogram_input.shape}")
                
                # Verify the shape matches what the model expects
                expected_shape = self.model.input_shape
                if spectrogram_input.shape[1:] == (128, 345, 1):
                    print("✅ Input shape matches model requirements!")
                else:
                    print(f"⚠️ Shape mismatch: got {spectrogram_input.shape[1:]}, expected (128, 345, 1)")
                
                # Make prediction with error handling for shape issues
                try:
                    probabilities = self.model.predict(spectrogram_input, verbose=0)[0]
                    prediction_idx = np.argmax(probabilities)
                    prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
                    
                    # Update results
                    self.prediction_label.config(text=f"🎭 {prediction.upper()}")
                    
                    # Update confidence bars
                    for i, emotion in enumerate(self.label_encoder.classes_):
                        confidence = probabilities[i] * 100
                        self.confidence_vars[emotion].set(f"{confidence:.1f}%")
                        
                        # Animate the progress bar
                        self.animate_confidence_bar(emotion, confidence)
                    
                    self.status_label.config(text="Analysis complete!")
                    
                except Exception as pred_error:
                    print(f"Prediction error: {pred_error}")
                    print(f"Input shape: {spectrogram_input.shape}")
                    print(f"Expected shape: {self.model.input_shape}")
                    
                    # Try to provide helpful error message
                    error_msg = (f"Model prediction failed.\n\n"
                               f"Input shape: {spectrogram_input.shape}\n"
                               f"Expected: {self.model.input_shape}\n\n"
                               f"This usually means the model was trained with different audio parameters.\n"
                               f"Please run the debug script first to diagnose the issue.")
                    
                    messagebox.showerror("Prediction Error", error_msg)
                    self.status_label.config(text="Prediction failed - check console for details")
                
            else:
                messagebox.showerror("Error", "Failed to process audio.")
                
        except Exception as e:
            print(f"Processing error: {e}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            self.reset_ui()
    
    def animate_confidence_bar(self, emotion, target_value):
        """Animate confidence bar to target value"""
        def update_bar(current_val, target_val, step=0):
            if step < 20:
                new_val = current_val + (target_val - current_val) * (step / 20)
                self.confidence_bars[emotion]['value'] = new_val
                self.root.after(50, lambda: update_bar(new_val, target_val, step + 1))
            else:
                self.confidence_bars[emotion]['value'] = target_val
        
        update_bar(0, target_value)
    
    def update_visualizations(self, waveform=None, spectrogram=None):
        """Update waveform and spectrogram visualizations"""
        # Clear previous plots
        self.ax_wave.clear()
        self.ax_spec.clear()
        
        if waveform is not None:
            # Plot waveform
            time_axis = np.linspace(0, self.DURATION, len(waveform))
            self.ax_wave.plot(time_axis, waveform, color='#3498db', linewidth=1)
            self.ax_wave.set_title('Waveform', color='white', fontsize=12)
            self.ax_wave.set_xlabel('Time (s)', color='white')
            self.ax_wave.set_ylabel('Amplitude', color='white')
            self.ax_wave.tick_params(colors='white')
            self.ax_wave.set_facecolor('#2c3e50')
            self.ax_wave.grid(True, alpha=0.3)
        else:
            # Empty waveform plot
            self.ax_wave.text(0.5, 0.5, 'No audio recorded', 
                            transform=self.ax_wave.transAxes,
                            ha='center', va='center', color='white', fontsize=14)
            self.ax_wave.set_facecolor('#2c3e50')
        
        if spectrogram is not None:
            # Plot spectrogram
            time_frames = np.linspace(0, self.DURATION, spectrogram.shape[1])
            mel_frequencies = range(spectrogram.shape[0])
            
            im = self.ax_spec.imshow(spectrogram, aspect='auto', origin='lower',
                                   extent=[0, self.DURATION, 0, len(mel_frequencies)],
                                   cmap='viridis')
            self.ax_spec.set_title('Mel Spectrogram', color='white', fontsize=12)
            self.ax_spec.set_xlabel('Time (s)', color='white')
            self.ax_spec.set_ylabel('Mel Frequency', color='white')
            self.ax_spec.tick_params(colors='white')
            self.ax_spec.set_facecolor('#2c3e50')
        else:
            # Empty spectrogram plot
            self.ax_spec.text(0.5, 0.5, 'No spectrogram available',
                            transform=self.ax_spec.transAxes,
                            ha='center', va='center', color='white', fontsize=14)
            self.ax_spec.set_facecolor('#2c3e50')
        
        # Refresh the canvas
        self.canvas.draw()
    
    def reset_ui(self):
        """Reset UI to initial state"""
        self.is_recording = False
        self.record_btn.config(text="🔴 Start Recording", bg='#e74c3c')
        self.progress['value'] = 0

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = VoiceEmotionGUI(root)
    
    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1200 // 2)
    y = (root.winfo_screenheight() // 2) - (800 // 2)
    root.geometry(f"1200x800+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()