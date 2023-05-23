import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = 'models/rps_model/gesture_recognizer.task'

VALID_OUTPUTS = ['paper','rock','scissors']

temp = ''

# This callback is called whenever the task has finished processing a video frame
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if not(len(result.gestures) > 0): return
    category_name = result.gestures[0][0].category_name
    
    global temp
    if category_name.lower() not in VALID_OUTPUTS or category_name == temp: return
    temp = category_name
    
    score = result.gestures[0][0].score
    print(f'Result: {category_name}         Score: {score:.2f}\n')

# Create a gesture recognizer instance with the live stream mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv2.VideoCapture(0)

# The detector is initialized. Use it here.
with GestureRecognizer.create_from_options(options) as RECOGNIZER:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) # The recognzer only accepts type int
        RECOGNIZER.recognize_async(mp_image, frame_timestamp_ms)
        
        # Show to screen
        cv2.imshow('Gesture-Recognizer: Rock, Paper, Scissors', cv2.flip(frame, 1))
        
        # Break and close camera feed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()