import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH, LOG_DIR, MODEL_PATH

def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(ACTIONS)}

    for action in ACTIONS:
        action_dir = os.path.join(DATA_PATH, action)
        if os.path.exists(action_dir) and os.path.isdir(action_dir):
            sequence_dirs = [d for d in os.listdir(action_dir) if d.isdigit()]
            for sequence in np.array(sequence_dirs).astype(int):
                window = []
                for frame_num in range(SEQUENCE_LENGTH):
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(ACTIONS)).astype(int)
    return X, y

def train_model(X, y):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(ACTIONS), activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    model.fit(X, y, epochs=2000, callbacks=[tb_callback])
    model.save(MODEL_PATH)

if __name__ == "__main__":
    X, y = load_data()
    train_model(X, y)
