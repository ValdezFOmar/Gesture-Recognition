import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt

# ASL Sign Language Gestures
DATASET_PATH = "asl_alphabet_train"

# Print labels
print(DATASET_PATH)
labels = []
for i in os.listdir(DATASET_PATH):
  if os.path.isdir(os.path.join(DATASET_PATH, i)):
    labels.append(i)
labels.sort()
print(labels)

# Load he dataset
data = gesture_recognizer.Dataset.from_folder(
    dirname=DATASET_PATH,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# Train the model
hparams = gesture_recognizer.HParams(export_dir="exported_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# Evaluate the model performance
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

# Export to Tensorflow Lite Model
model.export_model()