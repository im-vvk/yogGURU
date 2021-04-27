from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


PATH = os.getcwd()
cwd = os.getcwd()
NUM_CLASSES = 5

IMAGE_RESIZE = 224
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']


# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 5
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 10
BATCH_SIZE_VALIDATION = 10

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1


train_dir = cwd + "/data/yoga_dataset/images/train/"
validation_dir = cwd + "/data/yoga_dataset/images/test/"

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


image_size = IMAGE_RESIZE

data_generator = ImageDataGenerator()

# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE_TRAINING,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical')

test_generator = data_generator.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical')


def get_model(IMAGE_RESIZE=224):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(
        1./127.5, offset=-1)
    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(
        NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION)

    inputs = tf.keras.Input(shape=(IMAGE_RESIZE, IMAGE_RESIZE, 3))
    # x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # base_learning_rate = 0.0001
    # optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate)

    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])

    sgd = tf.keras.optimizers.SGD(
        lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

    return model


model = get_model()


cb_early_stopper = EarlyStopping(
    monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath='{cwd}/models/mbv2_best_1.h5',
                                  monitor='val_loss', save_best_only=True, mode='auto')

history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=STEPS_PER_EPOCH_VALIDATION,
    callbacks=[cb_checkpointer, cb_early_stopper]
)
