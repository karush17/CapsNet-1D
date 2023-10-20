"""Implements real time exeuction of sensor, model and GUI."""

from typing import Any

import argparse
import numpy as np
import matplotlib.pyplot as plt
import serial

from keras import models, optimizers
from pretraining.pretrain_capsnet import CapsNet

act_name = ['He needs Help', 
            'I want Water', "I don't want Water", 
            'He likes Bread', 'They like Bread', 
            'What are their names?', "What is your Father's name?", "What is the time?", 
            "Where is the Clinic?", "Wheres is the Bank?"]


json_file = open('cnn_cc50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = models.model_from_json(loaded_model_json)
cnn.load_weights("new_cnn.h5")
rmsprop = optimizers.RMSprop(lr=0.0001, decay=0)
cnn.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

def angles_extract(a_x, a_y, a_z, g_x, g_y, g_z) -> np.ndarray:
    """Extract angles from raw data stream."""
    angles = np.zeros((1,3))
    alpha = 0.85
    dt = 0.010
    for w in range(1,len(a_x)):
        angle_conc = np.zeros((0,0))
        acc_angle_y = np.arctan(-1*a_x[w]/(np.sqrt(a_y[w]**2 + a_z[w]**2)))
        acc_angle_x = np.arctan(-1*a_y[w]/(np.sqrt(a_x[w]**2 + a_z[w]**2)))
        acc_angle_z = 0

        if w == 1:
            last_angle_x = 0
            last_angle_y = 0
            last_angle_z = 0

        gyro_angle_x = g_x[w]*dt + last_angle_x
        gyro_angle_y = g_y[w]*dt + last_angle_y
        gyro_angle_z = g_z[w]*dt + last_angle_z

        angle_x = alpha*gyro_angle_x + (1-alpha)*acc_angle_x
        angle_y = alpha*gyro_angle_y + (1-alpha)*acc_angle_y
        angle_z = alpha*gyro_angle_z + (1-alpha)*acc_angle_z

        last_angle_x = angle_x
        last_angle_y = angle_y
        last_angle_z = angle_z

        angle_conc = [angle_x,angle_y,angle_z]
        angles = np.r_['0,2',angles,angle_conc]

    return angles

def test(model: Any, X_test: np.ndarray) -> None:
    """Evaluates the model on incoming signal."""
    prediction, _ = model.predict(X_test)
    print('CapsNet: ' + act_name[int(np.argmax(prediction, axis=1))])
    print(max(prediction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on Gestures.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0, type=float,
                        help="The value multiplied by lr at \
                            each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in \
                            routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most \
                            in each direction.")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. \
                            Should be specified when testing")
    args = parser.parse_args()

    X_train = np.load('11_data.npy')
    model, eval_model, manipulate_model = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=10,
                                                  routings=args.routings)
    eval_model.load_weights("new_capsnet30.h5")


    # Record and predict in real time
    plt.cla()
    ser = serial.Serial('COM3', 115200, bytesize = 8, stopbits = 1, timeout = 2)
    ser.flush()

    data = np.zeros((10,1))
    time = np.zeros((0,1))
    a_x=np.zeros((0,1))
    a_y=np.zeros((0,1))
    a_z=np.zeros((0,1))
    g_x=np.zeros((0,1))
    g_y=np.zeros((0,1))
    g_z=np.zeros((0,1))

    while True:
        try:
            data_stream = float(ser.readline().strip())
        except ValueError:
            data_stream = float(data[len(data) - 10])
        data = np.append(data, data_stream)
        if float(data_stream) == 555555:
            time = np.append(time, [float(data[-2])])
            a_z = np.append(a_z, [float(data[-3])])
            a_y = np.append(a_y, [float(data[-4])])
            a_x = np.append(a_x, [float(data[-5])])
            g_z = np.append(g_z, [float(data[-6])])
            g_y = np.append(g_y, [float(data[-7])])
            g_x = np.append(g_x, [float(data[-8])])
            if time[-1] in range(10000,10200):
                break

    ser.close()

    angles = angles_extract(a_x, a_y, a_z, g_x, g_y, g_z)
    a_x = np.expand_dims(np.transpose(a_x),axis=0)
    a_x = np.expand_dims(np.transpose(a_x),axis=0)
    a_y = np.expand_dims(np.transpose(a_y),axis=0)
    a_y = np.expand_dims(np.transpose(a_y),axis=0)
    a_z = np.expand_dims(np.transpose(a_z),axis=0)
    a_z = np.expand_dims(np.transpose(a_z),axis=0)
    g_x = np.expand_dims(np.transpose(g_x),axis=0)
    g_x = np.expand_dims(np.transpose(g_x),axis=0)
    g_y = np.expand_dims(np.transpose(g_y),axis=0)
    g_y = np.expand_dims(np.transpose(g_y),axis=0)
    g_z = np.expand_dims(np.transpose(g_z),axis=0)
    g_z = np.expand_dims(np.transpose(g_z),axis=0)
    angles = np.expand_dims(angles,axis=0)
    final = np.concatenate((a_x, a_y, a_z, g_x, g_y, g_z, angles), axis = 2)

    for k in range(1, 1033 - len(final[0, :, 1])):
        final = np.hstack((final, np.expand_dims(final[:, -1, :], axis = 0)))

    final = final[:, 0:1032, :]
    pred = cnn.predict(final)
    prediction, _ = eval_model.predict(final)
    test(eval_model, final)
