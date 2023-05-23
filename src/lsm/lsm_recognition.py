import os
import tensorflow as tf

assert tf.__version__.startswith("2")

from mediapipe_model_maker import gesture_recognizer

# Lenguaje de señas mexicano
DATASET_PATH = "lsm_train"
MODEL_PATH = "models/lsm_model"


def main():
    # Print labels
    print(DATASET_PATH)
    labels = []
    for i in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH, i)):
            labels.append(i)
    labels.sort()
    print("-----------------------------------------------------")
    print(labels)
    print("-----------------------------------------------------")

    # Load the dataset (~2GB aprox.)
    data = gesture_recognizer.Dataset.from_folder(
        dirname=DATASET_PATH, hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    train_data, res_data = data.split(0.8)
    validation_data, test_data = res_data.split(0.5)

    # Train the model
    # Adjust number of epochs
    hparams = gesture_recognizer.HParams(export_dir=MODEL_PATH, epochs=500)
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data, validation_data=validation_data, options=options
    )

    # Test the trained model
    loss, acc = model.evaluate(test_data, batch_size=1)
    print("-----------------------------------------------------")
    print(f"Test loss:{loss}, Test accuracy:{acc}")
    print("-----------------------------------------------------")

    # Export to Tensorflow Lite Model
    model.export_model()


# Script for training and exporting model
if __name__ == "__main__":
    main()
