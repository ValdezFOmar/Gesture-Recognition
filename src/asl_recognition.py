import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

# ASL Sign Language Gestures
DATASET_PATH = "asl_alphabet_train"
MODEL_PATH = "asl_model"

def main():
    # Print labels
    print(DATASET_PATH)
    labels = []
    for i in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH, i)):
            labels.append(i)
    labels.sort()
    print('--------------------------------------')
    print(labels)
    print('--------------------------------------')

    # Load the dataset (2GB aprox.)
    data = gesture_recognizer.Dataset.from_folder(
        dirname=DATASET_PATH,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    train_data, validation_data = data.split(0.8)

    # Train the model
    hparams = gesture_recognizer.HParams(export_dir=MODEL_PATH)
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    # Export to Tensorflow Lite Model
    model.export_model()

# Script for training and exporting model
if __name__ == 'main':
    main()