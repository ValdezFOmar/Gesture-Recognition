import mediapipe as mp
import os
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from matplotlib import pyplot as plt

# Script for testing the model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

TESTDATA_PATH = "asl_alphabet_test"
MODEL_PATH = "models/asl_model/gesture_recognizer.task"

# Create an GestureRecognizer object.
BASE_OPTIONS = python.BaseOptions(model_asset_path=MODEL_PATH)
OPTIONS = vision.GestureRecognizerOptions(base_options=BASE_OPTIONS)
RECOGNIZER = vision.GestureRecognizer.create_from_options(OPTIONS)  # Loaded model

SIGNS = ['A','B','C','D','F','G','H','I','J','K','L','M','N','O','P','Q',
        'R','S','T','U','V','W','X','Y','Z','none','space']

images = []
results = []

# Loads the test images
def get_test_images():
    for sign in SIGNS:
        path_to_image = os.path.join(TESTDATA_PATH, f'{sign}_test.jpg')
        image_file = mp.Image.create_from_file(path_to_image) # Load Image
        images.append(image_file)

# Gets the results from the loaded images using RECOGNIZER
def get_results():
    for image in images:
        recognition_result = RECOGNIZER.recognize(image)
        top_gesture = recognition_result.gestures[0][0]
        hand_landmarks = recognition_result.hand_landmarks
        results.append((top_gesture, hand_landmarks))

# Displays one image along with the predicted category name and score
def display_one_image(image, title, subplot, titlesize=16):
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title,
                  fontsize=int(titlesize),
                  color='black',
                  fontdict={'verticalalignment':'center'},
                  pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)

# Displays a batch of images with the gesture category and its score along with the hand landmarks
def display_images_with_results(images, results):
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def main():
    get_test_images()
    get_results()
    display_images_with_results()

if __name__ == '__main__':
    main()