import mediapipe as mp

from mediapipe_model_maker import gesture_recognizer 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# TODO.
# Load the test images and test the mode with them,
# use opencv to show the images, alongside their
# respective landmarks and the model prediction.

# NOTE. Check README file for the mediapipe guide

# Script for testing the model

TESTDATA_PATH = "asl_alphabet_test"
MODEL_PATH = "models/asl_model/gesture_recognizer.task"

# Create an GestureRecognizer object.
BASE_OPTIONS = python.BaseOptions(model_asset_path=MODEL_PATH)
OPTIONS = vision.GestureRecognizerOptions(base_options=BASE_OPTIONS)
RECOGNIZER = vision.GestureRecognizer.create_from_options(OPTIONS)  # Loaded model

# Functions and logic for testing....