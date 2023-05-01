from mediapipe_model_maker import gesture_recognizer 

TESTDATA_PATH = "asl_alphabet_test"

# TODO. Code for importing model once its exported.


def main():
    # Load test data
    test_data = gesture_recognizer.Dataset.from_folder(
        dirname=TESTDATA_PATH,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )

    # Evaluate the model performance
    loss, acc = model.evaluate(test_data, batch_size=1)
    print('--------------------------------------')
    print(f"Test loss:{loss}, Test accuracy:{acc}")
    print('--------------------------------------')
    
# Script for testing model
if __name__ == '__main__':
    main()