from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys

import os


class_name = sorted({'Bow_Pose_or_Dhanurasana_': 0,
                     'Bridge_Pose_or_Setu_Bandha_Sarvangasana_': 1,
                     'Cobra_Pose_or_Bhujangasana_': 2,
                     'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_': 3,
                     'Tree_Pose_or_Vrksasana_': 4}.keys())


def get_score_label(score):
    if score > 80:
        return "Pro"

    elif score > 65:
        return "Good"

    elif score > 50:
        return "Average"

    elif score > 30:
        return "Rookie"

    else:
        return "Try Again"


if len(sys.argv) != 2:
    msg = 'Error: Please Provide an Image Path in Command line'
    print("\033[91m {}\033[00m" .format(msg))
    exit()


cwd = os.getcwd()

model_path = cwd+"/models/mbv2_best.h5"

model = keras.models.load_model(model_path)


image_path = sys.argv[1]

IMG_SIZE = (224, 224)

image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)

input_arr = tf.keras.preprocessing.image.img_to_array(image)

input_arr = np.array([input_arr])  # Convert single image to a batch.

predictions = model.predict(input_arr)
idx = np.argmax(predictions)

cname = class_name[idx].split('_')

scr_label = get_score_label(int(predictions[0][idx]*100))

print(f"{cname[0]} {cname[1]}: {scr_label}")
