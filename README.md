# 🤖 Real-Time ASL Gesture Recognition

## 🎯 Project Goal

This project implements a system for real-time American Sign Language (ASL) gesture recognition using computer vision and deep learning. It leverages MediaPipe for robust hand and body pose estimation and an LSTM neural network built with TensorFlow/Keras for gesture classification.

## ✨ Core Features

* **Real-time gesture detection:** Recognizes predefined ASL gestures from a live webcam feed.
* **MediaPipe Integration:** Uses MediaPipe Holistic for comprehensive landmark detection (pose, face, hands).
* **LSTM Model:** Employs a Long Short-Term Memory (LSTM) network to learn temporal patterns in gesture sequences.
* **Modular Design:** Code is organized into separate scripts for configuration, data collection, model training, and prediction.

## 🛠️ Project Structure

* `config.py`: Contains global constants and configuration parameters for the project (e.g., actions to recognize, data paths, model paths).
* `mediapipe_detection.py`: Handles landmark detection and processing using the MediaPipe library. Includes functions for extracting keypoints and drawing landmarks.
* `data-collection.py`: Script to capture and save gesture data (sequences of keypoints) for training the model.
* `model-training.py`: Script to load the collected data, define, train, and save the LSTM model.
* `gesture-prediction.py`: Script to load the trained model and perform real-time gesture prediction from a webcam.
* `MP_Data/`: (Directory to be created by `data-collection.py`) Stores the collected gesture data as `.npy` files, organized by action and sequence.
* `Logs/`: (Directory to be created by `model-training.py`) Stores TensorBoard logs during model training.
* `action.h5`: (File created by `model-training.py`) The saved trained Keras model.

## ✅ Prerequisites

* Python 3.x
* OpenCV
* MediaPipe
* NumPy
* TensorFlow (or tensorflow-gpu)
* SciPy
* A webcam

## ✅ Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Modify `config.py` (if needed):**
    * Update `ACTIONS` with the gestures you want to recognize.
    * Ensure `DATA_PATH`, `LOG_DIR`, and `MODEL_PATH` point to valid locations.

## Usage

### 1. Data Collection

* Run the `data-collection.py` script to collect training data for your defined gestures.
    ```bash
    python data-collection.py
    ```
* The script will prompt you to perform each gesture for a set number of sequences and frames. Follow the on-screen instructions.
* Data will be saved in the `MP_Data` directory (or the path specified in `config.py`).

### 2. Model Training

* Once you have collected sufficient data, run the `model-training.py` script to train the LSTM model.
    ```bash
    python model-training.py
    ```
* This script will load the data, train the model, and save the trained model as `action.h5` (or the path specified in `config.py`).
* You can monitor training progress using TensorBoard:
    ```bash
    tensorboard --logdir=Logs
    ```

### 3. Gesture Prediction

* After training and saving the model, run the `gesture-prediction.py` script to perform real-time gesture recognition.
    ```bash
    python gesture-prediction.py
    ```
* This will open a webcam feed, and the recognized gestures will be displayed on the screen. Press 'q' to quit.

### (Optional) Testing MediaPipe Detection

* To test the MediaPipe landmark detection independently, you can run `mediapipe_detection.py`:
    ```bash
    python mediapipe_detection.py
    ```
    This will show your webcam feed with MediaPipe landmarks drawn. Press 'q' to quit.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.