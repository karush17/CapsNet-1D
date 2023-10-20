"""Implements the GEAR V2.0 GUI interface."""

import numpy as np
import matplotlib.pyplot as plt
import serial

from PyQt5 import QtCore, QtGui, QtWidgets
from keras import layers, models, optimizers
from caps_net.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import keras.backend as K

_translate = QtCore.QCoreApplication.translate
act_name = ['He needs Help', 
            'I want Water', "I don't want Water", 
            'He likes Bread', 'They like Bread', 
            'What are their names?', "What is your Father's name?", "What is the time?", 
            "Where is the Clinic?", "Where is the Bank?"]


## LOADING THE MODEL
json_file = open('cnn_cc50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = models.model_from_json(loaded_model_json)
cnn.load_weights("new_cnn.h5")
rmsprop = optimizers.RMSprop(lr=0.0001, decay=0)
cnn.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

def angles_extract(a_x,a_y,a_z,g_x,g_y,g_z):
    angles = np.zeros((1,3))
    alpha = 0.85
    dt = 0.010
    for w in range(1,len(a_x)):
        angle_conc = np.zeros((0,0))
        acc_angle_y = np.arctan(-1*a_x[w]/(np.sqrt(a_y[w]**2 + a_z[w]**2)))
        acc_angle_x = np.arctan(-1*a_y[w]/(np.sqrt(a_x[w]**2 + a_z[w]**2)))
        acc_angle_z = 0
        
        if (w==1):
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
    
def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=256, kernel_size=12, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=5, n_channels=20, kernel_size=12, strides=2, padding='valid')
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=10, num_routing=routings,
                             name='digitcaps')(primarycaps)
    
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps) 

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    dec_3 = decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid', input_dim=10*n_class))
#    dec_1 = decoder.add(layers.Dense(16, activation='relu', input_dim=10*n_class))
#    dec_2 = decoder.add(layers.Dense(16, activation='relu'))
#    decoder.add(layers.Dropout(0.40))
#    dec_3 = decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    dec_4 = decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked)]) #masked_by_y
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 10))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

if __name__ == "__main__":
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on Gestures.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    
    # define model
    X_train = np.load('11_data.npy')
    model, eval_model, manipulate_model = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=10,
                                                  routings=args.routings)
    eval_model.load_weights("new_capsnet30.h5")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(170, 170, 255);\n"
"")
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(356, 113, 75, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(25, 430, 271, 51))
        self.textBrowser_4.setObjectName("textBrowser_4")
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.textBrowser_4.setFont(font)
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_5.setGeometry(QtCore.QRect(470, 430, 271, 51))
        self.textBrowser_5.setObjectName("textBrowser_5")
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.textBrowser_5.setFont(font)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(140, 369, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(560, 370, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(380, 370, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(619, 123, 47, 13))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(681, 119, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(379, 410, 21, 131))
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(2)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setObjectName("line")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(198, 113, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: rgb(170, 170, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(-3, 72, 811, 20))
        self.line_2.setStyleSheet("color: rgb(255, 85, 0);")
        self.line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_2.setLineWidth(2)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setObjectName("line_2")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(16, 4, 81, 71))
        self.label_7.setStyleSheet("")
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(150, 24, 471, 31))
        font = QtGui.QFont()
        font.setFamily("Myanmar Text")
        font.setPointSize(18)
        font.setItalic(False)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("color: rgb(0, 85, 255);")
        self.label_9.setLineWidth(1)
        self.label_9.setObjectName("label_9")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(700, 4, 81, 71))
        self.label_8.setStyleSheet("")
        self.label_8.setObjectName("label_8")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(10, 555, 781, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("")
        self.label_10.setObjectName("label_10")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(26, 170, 731, 171))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(40, 113, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color: rgb(170, 170, 255);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(320, 529, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Courier")
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.Play)
        self.pushButton_2.clicked.connect(self.ImportFiles)
        pixmap = QtGui.QPixmap("amity.jpg")
        pixmap = pixmap.scaled(self.label_7.width(),self.label_7.height(),QtCore.Qt.KeepAspectRatio)
        self.label_7.setPixmap(pixmap)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        pixmap = QtGui.QPixmap("serb.jpg")
        pixmap = pixmap.scaled(self.label_8.width(),self.label_8.height(),QtCore.Qt.KeepAspectRatio)
        self.label_8.setPixmap(pixmap)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.pushButton_3.clicked.connect(self.Description)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GEAR2.1"))
        self.pushButton.setText(_translate("MainWindow", "Play"))
        self.label_2.setText(_translate("MainWindow", "CNN"))
        self.label_3.setText(_translate("MainWindow", "CapsNet"))
        self.label_4.setText(_translate("MainWindow", "Vs"))
        self.label_5.setText(_translate("MainWindow", "Status: "))
        self.label_6.setText(_translate("MainWindow", "-"))
        self.pushButton_2.setText(_translate("MainWindow", "Import Files"))
        self.label_7.setText(_translate("MainWindow", "TextLabel"))
        self.label_9.setText(_translate("MainWindow", "Gesture Evaluator And Recognizer (GEAR2.1)"))
        self.label_8.setText(_translate("MainWindow", "TextLabel"))
        self.label_10.setText(_translate("MainWindow", "Funded by Science and Engineering Research Board, a statutory body of the Department of Science and Technology(DST), Government of India, SERB File No.- ECR/2016/000637. "))
        self.pushButton_3.setText(_translate("MainWindow", "Description"))
        self.label_11.setText(_translate("MainWindow", "Crafted by Karush"))

    
    def Description(self):
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(False)
        self.label.setFont(font)
        self.label.setText(_translate("MainWindow", "GEAR2.1 recognizes sentences from Indian Sign Language by making AIs compete with each other.\n\nTo play the Game follow these simple steps-\n1. Click on 'Import' to import the AI files.\n2. Select the sentence you wish to perform from the sentence list.\n3. Click on 'Play' and perform the sentence with gaps between each gestures.\n4. When finished, keep your hand in the final position."))
        
    def Play(self):
        #ser.close()
        ser = serial.Serial('COM3',115200,bytesize=8,stopbits=1,timeout=2)
        ser.flush()
        data = np.zeros((10,1));time = np.zeros((0,1))
        a_x=np.zeros((0,1));a_y=np.zeros((0,1));a_z=np.zeros((0,1))
        g_x=np.zeros((0,1));g_y=np.zeros((0,1));g_z=np.zeros((0,1))
        
        while 1:
            try:
                arduinoData = float(ser.readline().strip())
            except ValueError:
                arduinoData = float(data[len(data)-10])
            data = np.append(data,arduinoData)
            if float(arduinoData) == 555555:
                time = np.append(time,[float(data[-2])])
                a_z = np.append(a_z,[float(data[-3])])
                a_y = np.append(a_y,[float(data[-4])])
                a_x = np.append(a_x,[float(data[-5])])
                g_z = np.append(g_z,[float(data[-6])])
                g_y = np.append(g_y,[float(data[-7])])
                g_x = np.append(g_x,[float(data[-8])])
                if time[-1] in range(10000,10200):
                    break
                
        ser.close()
        
        fig, ax1 = plt.subplots()
        plt.grid(True)
        ax1.plot(time, g_x, 'r-')
        ax1.plot(time, g_y, 'g-')
        ax1.plot(time, g_z,'y-')
        plt.savefig('plot.jpg')
        pixmap = QtGui.QPixmap("plot.jpg")
        pixmap = pixmap.scaled(self.label.width(),self.label.height())#QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        
        
        angles = angles_extract(a_x,a_y,a_z,g_x,g_y,g_z)
        a_x = np.expand_dims(np.transpose(a_x),axis=0);a_x = np.expand_dims(np.transpose(a_x),axis=0)
        a_y = np.expand_dims(np.transpose(a_y),axis=0);a_y = np.expand_dims(np.transpose(a_y),axis=0)
        a_z = np.expand_dims(np.transpose(a_z),axis=0);a_z = np.expand_dims(np.transpose(a_z),axis=0)
        g_x = np.expand_dims(np.transpose(g_x),axis=0);g_x = np.expand_dims(np.transpose(g_x),axis=0)
        g_y = np.expand_dims(np.transpose(g_y),axis=0);g_y = np.expand_dims(np.transpose(g_y),axis=0)
        g_z = np.expand_dims(np.transpose(g_z),axis=0);g_z = np.expand_dims(np.transpose(g_z),axis=0)
        angles = np.expand_dims(angles,axis=0)
        final = np.concatenate((a_x,a_y,a_z,g_x,g_y,g_z,angles),axis=2)
        
        for k in range(1,1033-len(final[0,:,1])):
            final = np.hstack((final,np.expand_dims(final[:,-1,:],axis=0)))
        
        final = final[:,0:1032,:]
        pred = cnn.predict(final)
        #print('CNN: ' + act_name[int(np.argmax(pred, axis=1))])
        self.textBrowser_4.setText(_translate("MainWindow",act_name[int(np.argmax(pred, axis=1))]))
        prediction,x_recon = eval_model.predict(final)
        self.textBrowser_5.setText(_translate("MainWindow",act_name[int(np.argmax(prediction, axis=1))]))
        self.label_6.setText(_translate("MainWindow", "Done!"))



    def ImportFiles(self):
        self.label_6.setText(_translate("MainWindow", "Imported!"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
