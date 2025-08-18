Voice Emotion Detection System
This project is a real-time voice emotion detection system that uses a Convolutional Neural Network (CNN) to classify emotions (happy, sad, angry) from live audio.

1. What's Included
app.py: The main application for training the model and running real-time predictions in the terminal.

emotion.gui.py: A tool to record your voice and see a live prediction along with a waveform and spectrogram visualization.

sort_ravdess.py: A helper script to automatically sort the RAVDESS dataset (optional).

/models/: This folder contains the pre-trained CNN model (emotion_cnn_model.h5) and the label encoder.

/features/: This folder contains the processed spectrogram data used for training.

requirements.txt: A list of all the Python libraries needed to run the project.

2. Setup Instructions for Your Friend
Follow these steps to get the project running on a new computer.

Step A: Install Python
Make sure you have Python installed. You can download it from python.org.

Step B: Install Required Libraries
Open your terminal or command prompt.

Navigate to the main project folder (the one containing this README file).

Run the following command to install all the necessary libraries at once:

pip install -r requirements.txt

3. How to Run the Project
You have two ways to use this system:

->Run the Prediction with Visualization
python emotion_gui.py {make sure you are in the directory}


->Run Prediction only without waveform
python app.py