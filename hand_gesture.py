# from google.colab import files
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt

DATASET_PATH = "rps_data_sample"

# Print labels
print(DATASET_PATH)
labels = []
for i in os.listdir(DATASET_PATH):
  if os.path.isdir(os.path.join(DATASET_PATH, i)):
    labels.append(i)
print(labels)

# example images for each gesture
# NUM_EXAMPLES = 5

# for label in labels:
#   label_dir = os.path.join(dataset_path, label)
#   example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
#   fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
#   for i in range(NUM_EXAMPLES):
#     axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
#     axs[i].get_xaxis().set_visible(False)
#     axs[i].get_yaxis().set_visible(False)
#   fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')

# plt.show()

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
