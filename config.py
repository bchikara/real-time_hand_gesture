import os
import numpy as np

# Define the actions as a global constant
ACTIONS = np.array(['hello', 'thanks', 'eat food'])

# Define other constants
DATA_PATH = 'MP_Data'  # Modify this to the actual data path
SEQUENCE_LENGTH = 30
LOG_DIR = 'Logs'
MODEL_PATH = 'action.h5'
