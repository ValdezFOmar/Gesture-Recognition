import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = 'models/asl_model/gesture_recognizer.task'

default_value = ('none',0)
prediction = default_value
temp = ''

# This callback is called whenever the task has finished processing a video frame
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global prediction
    global temp
    
    if not(len(result.gestures) > 0):
        prediction = default_value
        return
    
    category_name = result.gestures[0][0].category_name
    score = result.gestures[0][0].score
    
    # if category_name == temp: return
    # temp = category_name
    
    # print(f'Result: {category_name}   Score: {score:.2f}\n')
    prediction = (category_name, score)

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
        
        text = f'Sign: {prediction[0]}      Score: {prediction[1]:.2f}'
        
        fliped_frame = cv2.flip(frame, 1)
        
        # Show to screen
        cv2.putText(img=fliped_frame,
                    text=text,
                    org=(100,50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=3)
        
        cv2.imshow('Gesture-Recognizer: American Sign language (ASL)', fliped_frame)
        
        # Break and close camera feed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()