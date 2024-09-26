from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from ultralytics import YOLO
import msg as mn

train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

train_data=train.flow_from_directory('C:\\Users\\barsh\Desktop\dataa\\dataa\\training',target_size=(200,200),batch_size=5,class_mode='binary')
validation_data=validation.flow_from_directory('C:\\Users\\barsh\Desktop\\dataa\\dataa\\valid',target_size=(200,200),batch_size=5,class_mode='binary')

model1=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Flatten(),
    #
    tf.keras.layers.Dense(512,activation='relu'),
    #
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model1.compile(loss='binary_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['accuracy'])

model2=YOLO('yolov8n.pt')

model_fit = model1.fit(train_data, steps_per_epoch=3, epochs=25, validation_data=validation_data)



dir_path = 'C:\\Users\\barsh\\Desktop\\dataa\\dataa\\testing'
accident_folder = 'C:\\Users\\barsh\\ditected_res'

if not os.path.exists(accident_folder):
    os.makedirs(accident_folder)

for i in os.listdir(dir_path):
    img_path = os.path.join(dir_path, i)

    img = cv2.imread(img_path)

    results = model2(img_path)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy.int().tolist()

            if len(xyxy) == 4:
                x, y, x_end, y_end = xyxy
            elif len(xyxy) == 1 and len(xyxy[0]) == 4:
                x, y, x_end, y_end = xyxy[0]
            else:
                print(f"Unexpected bounding box format: {xyxy}")
                continue

            cv2.rectangle(img, (x, y), (x_end, y_end), (0, 255, 0), 2)

    img_custom = image.load_img(img_path, target_size=(200, 200))
    X_custom = image.img_to_array(img_custom)
    X_custom = np.expand_dims(X_custom, axis=0)
    images_custom = np.vstack([X_custom])
    val = model1.predict(images_custom)
    if val == 0:
        prediction_text = "Non-accident"
    else:
        prediction_text = "Accident"

    cv2.putText(img, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if val == 1: 
        cv2.imwrite(os.path.join(accident_folder, i), img)  
        mn.send_msg()
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()