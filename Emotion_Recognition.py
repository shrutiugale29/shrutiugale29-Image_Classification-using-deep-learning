#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#To run the created .ipy file in your Jupyter Notebook, use the %run magic command:
#%run D:/your_folder/example_script.ipy 


# In[1]:


import os

# Change directory to the desired folder on D drive
os.chdir('D:\\IIT_Internship_Project')  # Replace 'your_folder' with the actual folder name or path

# Verify the current working directory
print(f"Current working directory: {os.getcwd()}")

# Content for the IPython file
content = """
# This is an example IPython script
print("Hello, this is an IPython script!")
"""

# Write content to a .ipy file in the new directory
file_path = 'Emotion_Recognition.ipy'
with open(file_path, 'w') as file:
    file.write(content)

print(f"IPython script saved at: {file_path}")


# In[2]:


get_ipython().system('pip install opencv-python')


# In[ ]:


## Setup


# In[3]:


import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# In[ ]:


## EXtracting Frames


# In[4]:


import os
import cv2
import numpy as np

# Function to extract frames at 3-second intervals and save them to output directory
def extract_frames_from_directory(video_dir, output_dir, interval=3):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frames_count = 0
    total_frames = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_path = os.path.join(root, file)
                frames = extract_frames(video_path, output_dir, interval)
                frames_count += len(frames)
                total_frames.extend(frames)
    
    return total_frames

# Function to extract frames from a single video file
def extract_frames(video_path, output_dir, interval=3):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video: {video_path}, FPS: {fps}, Frame count: {frame_count}, Duration: {duration} seconds")

    frame_number = 0
    for sec in range(0, int(duration), interval):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if success:
            frame_number += 1
            frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            print(f"Frame {frame_number} extracted at {sec} seconds and saved to {frame_path}")
        else:
            print(f"Failed to extract frame at {sec} seconds from {video_path}")
            break
    
    cap.release()
    return frames

# Example usage with your directory path
video_dir = r'D:\IIT_Internship_Project\DAiSEE\DataSet\Test'  # Replace with your video directory path
output_directory = r'D:\IIT_Internship_Project\DAiSEE\extracted_frames_Test'  # Replace with your desired output directory

extracted_frames = extract_frames_from_directory(video_dir, output_directory)
print(f"Total number of frames extracted: {len(extracted_frames)}")


# In[5]:


import os
import cv2
import numpy as np

# Function to extract frames at 3-second intervals and save them to output directory
def extract_frames_from_directory(video_dir, output_dir, interval=3):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frames_count = 0
    total_frames = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_path = os.path.join(root, file)
                frames = extract_frames(video_path, output_dir, interval)
                frames_count += len(frames)
                total_frames.extend(frames)
    
    return total_frames

# Function to extract frames from a single video file
def extract_frames(video_path, output_dir, interval=3):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video: {video_path}, FPS: {fps}, Frame count: {frame_count}, Duration: {duration} seconds")

    frame_number = 0
    for sec in range(0, int(duration), interval):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if success:
            frame_number += 1
            frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            print(f"Frame {frame_number} extracted at {sec} seconds and saved to {frame_path}")
        else:
            print(f"Failed to extract frame at {sec} seconds from {video_path}")
            break
    
    cap.release()
    return frames

# Example usage with your directory path
video_dir = r'D:\IIT_Internship_Project\DAiSEE\DataSet\Train'  # Replace with your video directory path
output_directory = r'D:\IIT_Internship_Project\DAiSEE\extracted_frames_Train'  # Replace with your desired output directory

extracted_frames = extract_frames_from_directory(video_dir, output_directory)
print(f"Total number of frames extracted: {len(extracted_frames)}")


# In[6]:


import os
import cv2
import numpy as np

# Function to extract frames at 3-second intervals and save them to output directory
def extract_frames_from_directory(video_dir, output_dir, interval=3):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frames_count = 0
    total_frames = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_path = os.path.join(root, file)
                frames = extract_frames(video_path, output_dir, interval)
                frames_count += len(frames)
                total_frames.extend(frames)
    
    return total_frames

# Function to extract frames from a single video file
def extract_frames(video_path, output_dir, interval=3):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video: {video_path}, FPS: {fps}, Frame count: {frame_count}, Duration: {duration} seconds")

    frame_number = 0
    for sec in range(0, int(duration), interval):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if success:
            frame_number += 1
            frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            print(f"Frame {frame_number} extracted at {sec} seconds and saved to {frame_path}")
        else:
            print(f"Failed to extract frame at {sec} seconds from {video_path}")
            break
    
    cap.release()
    return frames

# Example usage with your directory path
video_dir = r'D:\IIT_Internship_Project\DAiSEE\DataSet\Validation'  # Replace with your video directory path
output_directory = r'D:\IIT_Internship_Project\DAiSEE\extracted_frames_Validation'  # Replace with your desired output directory

extracted_frames = extract_frames_from_directory(video_dir, output_directory)
print(f"Total number of frames extracted: {len(extracted_frames)}")


# In[ ]:


## Load EfficientNet and Prepare for Fine-Tuning


# In[1]:


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load EfficientNetB0 model for feature extraction
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

# Create the final model for fine-tuning
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()


# In[ ]:


## Preprocess Frames


# In[1]:


## preprocess
import cv2
import os
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Function to preprocess frames from multiple folders and save them to specified output directory
def preprocess_frames_and_save(dataset_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_counts = {}
    for dataset_type, folder_path in dataset_paths.items():
        frames = load_frames_from_folder(folder_path)
        dataset_counts[dataset_type] = len(frames)
        for i, frame in enumerate(frames):
            preprocessed_frame = preprocess_frame(frame)
            save_path = os.path.join(output_dir, f"{dataset_type}_frame_{i}.jpg")
            cv2.imwrite(save_path, preprocessed_frame[:, :, ::-1])  # Save with BGR format
    return dataset_counts

# Function to load frames from a folder
def load_frames_from_folder(folder_path):
    frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frame_path = os.path.join(folder_path, filename)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
    return frames

# Function to preprocess a single frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = img_to_array(frame)            # Convert frame to NumPy array
    frame = np.expand_dims(frame, axis=0)  # Expand dimensions to create batch-like structure (1, 224, 224, 3)
    frame = preprocess_input(frame)        # Preprocess input according to EfficientNetB0 requirements
    return frame[0]  # Remove batch dimension

# Example usage:
dataset_paths = {
    'train': r'D:\IIT_Internship_Project\DAiSEE\extracted_frames_train',
    'test': r'D:\IIT_Internship_Project\DAiSEE\extracted_frames_test',
    'validation': r'D:\IIT_Internship_Project\DAiSEE\extracted_frames_validation'
}
output_directory = r'D:\IIT_Internship_Project\DAiSEE\preprocessed_frames'  # Specify your desired output directory here

# Preprocess frames from all datasets and save them to specified output directory
dataset_counts = preprocess_frames_and_save(dataset_paths, output_directory)

# Print number of frames processed for each dataset
for dataset_type, count in dataset_counts.items():
    print(f"Number of frames preprocessed for {dataset_type}: {count}")


# In[2]:


import cv2
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Function to preprocess a single frame according to EfficientNetB0 requirements
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))        # Resize frame to 224x224 pixels
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame from BGR to RGB
    frame = img_to_array(frame)                  # Convert frame to NumPy array
    frame = np.expand_dims(frame, axis=0)        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    frame = preprocess_input(frame)              # Preprocess input according to EfficientNetB0 requirements
    return frame

# Function to predict emotion for each frame using a pre-trained EfficientNetB0 model
def predict_emotion(frames, model):
    emotions = ['Engagement', 'Confusion', 'Frustration', 'Boredom', 'Others']
    predictions = []
    for frame in frames:
        try:
            preprocessed_frame = preprocess_frame(frame)
            preds = model.predict(preprocessed_frame)
            predicted_class = emotions[np.argmax(preds)]
            predictions.append(predicted_class)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            predictions.append("Error")
    return predictions

# Function to load frames from a list of paths
def load_frames_from_paths(frame_paths):
    frames = []
    for path in frame_paths:
        try:
            if os.path.exists(path):
                frame = cv2.imread(path)
                if frame is not None:
                    frames.append(frame)
                else:
                    print(f"Failed to load frame at path: {path}")
            else:
                print(f"File not found: {path}")
        except Exception as e:
            print(f"Error loading frame: {str(e)}")
    return frames

# Example frames with specific paths
frame_paths = [
    r"D:\IIT_Internship_Project\DAiSEE\preprocessed_frames\test_frame_133.jpg",
    r"D:\IIT_Internship_Project\DAiSEE\preprocessed_frames\test_frame_186.jpg",
    r"D:\IIT_Internship_Project\DAiSEE\preprocessed_frames\test_frame_247.jpg",
]

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions_layer = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions_layer)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load frames from file paths
frames = load_frames_from_paths(frame_paths)

# Predict emotions for the loaded frames
predictions = predict_emotion(frames, model)

# Print predictions
for prediction in predictions:
    print(prediction)


# In[ ]:


import cv2


# In[ ]:


pip install pillow


# In[ ]:


import cv2
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Function to preprocess a single frame according to EfficientNetB0 requirements
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))        # Resize frame to 224x224 pixels
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame from BGR to RGB
    frame = img_to_array(frame)                  # Convert frame to NumPy array
    frame = np.expand_dims(frame, axis=0)        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    frame = preprocess_input(frame)              # Preprocess input according to EfficientNetB0 requirements
    return frame

# Function to predict emotion for each frame using a pre-trained EfficientNetB0 model
def predict_emotion(frames, model):
    emotions = ['Engagement', 'Confusion', 'Frustration', 'Boredom', 'Others']
    predictions = []
    for frame in frames:
        preprocessed_frame = preprocess_frame(frame)
        preds = model.predict(preprocessed_frame)
        predicted_class = emotions[np.argmax(preds)]
        predictions.append(predicted_class)
    return predictions

# Function to load frames from a folder using OpenCV and fallback to PIL
def load_frames_from_folder(folder_path):
    frames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            frame_path = os.path.join(folder_path, filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                try:
                    # Fallback to PIL if OpenCV fails to read the image
                    with Image.open(frame_path) as img:
                        frame = np.array(img)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert PIL image to BGR format
                        frames.append(frame)
                        print(f"PIL successfully loaded the image: {frame_path}")
                except (UnidentifiedImageError, OSError) as e:
                    print(f"Failed to load frame at path: {frame_path}, Error: {e}")
            else:
                frames.append(frame)
                print(f"OpenCV successfully loaded the image: {frame_path}")
    return frames

# Example usage:
input_directory = r'D:\IIT_Internship_Project\DAiSEE\preprocessed_frames'  # Input directory with preprocessed frames

# Load frames from the input directory
frames = load_frames_from_folder(input_directory)

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions_layer = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions_layer)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Predict emotions for the loaded frames
if frames:
    predictions = predict_emotion(frames, model)

    # Print predictions
    for prediction in predictions:
        print(prediction)
else:
    print("No frames were successfully loaded. Please check the paths and the files.")


# In[ ]:




