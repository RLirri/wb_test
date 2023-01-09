import random
import math
from vehicle import Driver
from controller import Camera, Device, Display, GPS, Keyboard, Lidar, Robot, Supervisor
from typing import Any, List, Sequence, Tuple
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow import keras
import tensorflow as tf
import copy
import numpy as np
import time
import cv2
import os

layer_output_numbers = 676
number_of_classes = 80

weightsPath = r""
configPath = r""
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


def detect_all(pic):
    labelsPath = r""
    LABELS = None
    with open(labelsPath, 'rt') as f:
        LABELS = f.read().rstrip('\n').split("\n")

    np.random.seed(12)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    image = pic
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    return layerOutputs


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(256 / 8, activation='relu', input_shape=(182,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


def run_model(features: np.ndarray,
              model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    output = model(features)
    return output


model = create_model()

model.load_weights(r'')


def create_model_5():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(
            layer_output_numbers*(5+number_of_classes)+180+2,)),
        #keras.layers.Dense(256, activation='relu', input_shape=(182,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


model_5 = create_model_5()
model_0 = create_model_5()
model_1 = create_model_5()
model_2 = create_model_5()


driver = Driver()

timestep = int(driver.getBasicTimeStep())
TIME_STEP = 200
UNKNOWN = 99999.99
KP = 0.25
KI = 0.006
KD = 2

PID_need_reset = False
FILTER_SIZE = 3

enable_collision_avoidance = False
enable_display = False
has_gps = False
has_camera = False

camera = driver.getDevice("camera")
camera.width = -1
camera.height = -1
camera.fov = -1.0

sick = driver.getDevice("Sick LMS 291")
sick_width = -1
sick_range = -1.0
sick_fov = -1.0

display = driver.getDevice("display")
display_width = 0
display_height = 0
speedometer_image = None

gps = driver.getDevice("gps")
gps_coords = [0.0, 0.0, 0.0]
gps_speed = 0.0

speed = 0.0
steering_angle = 0.0
autodrive = True


def set_speed(kmh):
    # max speed
    if (kmh > 250.0):
        kmh = 250.0

    speed = kmh

    print("setting speed to " + str(kmh) + " km/h\n")
    driver.setCruisingSpeed(kmh)
    return 0


def print_help():
    print("You can drive this car!\n")
    print("Select the 3D window and then use the cursor keys to:\n")
    print("[LEFT]/[RIGHT] - steer\n")
    print("[UP]/[DOWN] - accelerate/slow down\n")
    return 0


def set_steering_angle(wheel_angle, steering_angle):
    if (wheel_angle - steering_angle > 0.1):
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle < -0.1):
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
    if (wheel_angle > 0.5):
        wheel_angle = 0.5
    elif (wheel_angle < -0.5):
        wheel_angle = -0.5
    driver.setSteeringAngle(wheel_angle)
    return 0


def compute_gps_speed():
    coords = gps.getValues()
    speed_ms = gps.getSpeed()
    gps_speed = speed_ms * 3.6  # convert from m/s to km/h
    return gps_speed


camera.width = camera.getWidth()
camera.height = camera.getHeight()
camera.fov = camera.getFov()
camera.enable(1)

sick = driver.getDevice("Sick LMS 291")
sick.enable(TIME_STEP)
sick_width = sick.getHorizontalResolution()
sick_range = sick.getMaxRange()
sick_fov = sick.getFov()

gps = driver.getDevice("gps")
gps.enable(TIME_STEP)


model_0.load_weights('./weights_0.h5')
model_1.load_weights('./weights_1.h5')
model_2.load_weights('./weights_2.h5')
hold_t = 0
action_taken = 1
data = [[]]
set_speed(30)
print_help()
t = 0
pre_route = random.random()


while driver.step() != -1:
    if t > 5000 or gps.getSpeed() < 6 and t > 250:
        # process data
        processed_data = copy.deepcopy(data)
        GAMMA = 0.9
        for i in range(len(processed_data)):
            # last_time_speed = 0
            last_time_steer_angle = 0
            for ii in range(len(data[i])):
                index = len(data[i]) - ii - 1
                processed_data[i][index] = []

                if ii == 0:
                    maxQ = -100/0.6
                else:
                    maxQ = processed_data[i][index + 1][0]

                steer = 1
                if ii == 0:
                    steer = 1
                else:
                    if data[i][index][181+layer_output_numbers*(5+number_of_classes)] < -0.15:
                        steer = 0
                    elif data[i][index][181+layer_output_numbers*(5+number_of_classes)] > 0.15:
                        steer = 2
                #processed_data[i][index].append(data[i][index][180+layer_output_numbers*(5+number_of_classes)] + GAMMA * maxQ)
                processed_data[i][index].append(GAMMA * maxQ)
                processed_data[i][index].append(steer)

        processed_data[0][0][0] = 0
        data[0][0][180+layer_output_numbers*(5+number_of_classes)] = 0
        # print(processed_data)

        # writing
        with open(r"round.txt", "r+", encoding="utf8") as f:
            rslist = []
            for i in f.readlines():
                rslist.append(i.strip())
        with open(r"round.txt", "w", encoding="utf8") as f:
            new_count = int(rslist[0])
            new_count += 1
            f.write(str(new_count))
        f.close()

        with open(r"C:\data.txt", "a", encoding="utf8") as f:
            for i in range(len(data)):
                for ii in range(len(data[i])):
                    text = ""
                    for iii in data[i][ii]:
                        text = text + str(iii) + ","
                    text = text + str(processed_data[i][ii][0]) + ","
                    text = text + str(processed_data[i][ii][1])
                    f.write(text+'\n')  # \n 换行符
        f.close()

        # training
        if new_count >= 10:

            with open(r"C:\data.txt", "r", encoding="utf8") as f:
                rslist = []
                for i in f.readlines():
                    rslist.append(i.strip())
            f.close()

            batch_size = 1000
            if len(rslist) < 1000:
                batch_size = int(len(rslist)*0.8)

            batch_data = []
            for i in rslist:
                list1 = i.split(',')
                record = []
                for ii in list1:
                    record.append(float(ii))
                batch_data.append(record)

            total_loss = 999999
            training_round = 0
            while training_round < 50 and total_loss/batch_size > 2:
                training_round += 1
                random.shuffle(batch_data)
                total_loss = 0
                for i in range(batch_size):

                    c = np.array([np.concatenate([batch_data[i][0:2], batch_data[i]
                                 [2:layer_output_numbers*(5+number_of_classes)+180+2]], axis=0)])

                    if int(batch_data[i][layer_output_numbers*(5+number_of_classes)+180+2+1]) == 0:
                        model_5 = model_0
                    elif int(batch_data[i][layer_output_numbers*(5+number_of_classes)+180+2+1]) == 1:
                        model_5 = model_1
                    elif int(batch_data[i][layer_output_numbers*(5+number_of_classes)+180+2+1]) == 2:
                        model_5 = model_2

                    with tf.GradientTape() as tape:
                        print(len(batch_data[0]))
                        y_ = run_model(c, model_5)[0]
                        y = y_.numpy()

                        y[0] = batch_data[i][layer_output_numbers *
                                             (5+number_of_classes)+180+2]
                        total_loss += abs(y[0] - y_[0])
                        loss = tf.keras.losses.MSE(y_true=y, y_pred=y_)
                        grads = tape.gradient(
                            loss, model_5.trainable_variables)
                        optimizer = tf.keras.optimizers.Adam(
                            learning_rate=0.001)
                        optimizer.apply_gradients(
                            zip(grads, model_5.trainable_variables))

            model_0.save_weights('./weights_0.h5')
            model_1.save_weights('./weights_1.h5')
            model_2.save_weights('./weights_2.h5')

            with open(r"round.txt", "w", encoding="utf8") as f:
                f.write(str(0))
            f.close()

            with open(r"C:\data.txt", "w", encoding="utf8") as f:
                f.close()

        print("reset")
        driver.simulationRevert()
        set_speed(50)
        t = 0

    if (t % (int)(TIME_STEP / driver.getBasicTimeStep()) == 0):
        sick_data = sick.getRangeImage()

        for i in range(len(sick_data)):
            if math.isinf(sick_data[i]):
                sick_data[i] = 80

        sick_data.append(compute_gps_speed())
        steering_angle = driver.getSteeringAngle()
        sick_data.append(steering_angle)
        camera.saveImage("a.png", 100)
        camera_image = cv2.imread("a.png")
        yolo0 = detect_all(camera_image)
        a = []
        #
        for iii in yolo0[2]:
            for iv in iii[0:(5+number_of_classes)]:
                a.append(iv)
        b = np.array(
            [tf.concat([sick_data[0:180], sick_data[180:182]], axis=0)])
        c = np.array([tf.concat([a, b[0]], axis=0)])
        #c = b
        if math.isnan(c[0][layer_output_numbers*(5+number_of_classes)+180+2-2]):
            c[0][layer_output_numbers*(5+number_of_classes)+180+2-2] = 0
        data[0].append(c[0])
        output0 = run_model(c, model_0)[0]
        output1 = run_model(c, model_1)[0]
        output2 = run_model(c, model_2)[0]
        output_all = []
        output_all.append(output0)
        output_all.append(output1)
        output_all.append(output2)
        print(output_all)
        if hold_t <= t:
            action_taken = np.argmax(output_all)
        if random.random() < 0.06:
            action_taken = action_taken + 1
            if action_taken == 3:
                action_taken = 0
            hold_t = t + 0
            print("explore")

        #
        if pre_route < 0.4:
            if t <= 200:
                action_taken = 1

            if t > 200 and t < 1100:
                action_taken = 1

            if t > 620 and t < 720:
                action_taken = 2

        if action_taken == 1:
            set_steering_angle(0, steering_angle)
        elif action_taken == 0:
            set_steering_angle(-0.3, steering_angle)
        elif action_taken == 2:
            set_steering_angle(0.3, steering_angle)

        print(action_taken)
    t += 1
