"""Pretrains a capsnet."""

from typing import Tuple, Any

import argparse
import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from caps_net.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


def CapsNet(input_shape: tf.tensor, n_class: int,
            routings: int) -> Tuple[Any, Any, Any]:
    """Implements the capsnet.
    
    Args:
        input_shape: shape of input dimension.
        n_class: number of classes.
        routings: number of layer routings.
    
    Returns:
        train_model: trained model.
        eval_model: evaluation model.
        manipulate_model: model with noise.
    """
    x = layers.Input(shape=input_shape)

    conv1 = layers.Conv1D(filters=256, kernel_size=12, strides=1,
                          padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=5, n_channels=20,
                             kernel_size=12, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=10,
                             num_routing=routings,
                             name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps) 
    y = layers.Input(shape=(n_class,))
    masked = Mask()(digitcaps)
    decoder = models.Sequential(name='decoder')

    train_model = models.Model([x, y], [out_caps, decoder(masked)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    noise = layers.Input(shape=(n_class, 10))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true: tf.tensor, y_pred: tf.tensor) -> tf.tensor:
    """Implements the soft margin loss.
    
    Args:
        y_true: true labels.
        y_pred: predicted labels.
    
    Returns:
        loss: calcuated loss.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def train(model: Any, X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray, y_test: np.ndarray):
    """Trains the model.
    
    Args:
        model: model to be trained.
        X_train: train dataset.
        y_train: train labels.
        X_test: test dataset.
        y_test: test labels.
    """
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True),
                  loss=[margin_loss, 'categorical_crossentropy'],
                  loss_weights=[1, args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    model.load_weights('new_capsnet20.h5')
    model.fit([X_train, y_train], [y_train, X_train],
              batch_size=args.batch_size, epochs=args.epochs, shuffle=True)
    model.save_weights('new_capsnet30.h5')
    return model


def test(model: Any, X_test: np.ndarray,
         y_test: np.ndarray) -> np.ndarray:
    """Tests the model on a test dataset.
    
    Args:
        model: model to be trained.
        X_test: test dataset.
        y_test: test labels.    
    """
    y_pred, _ = model.predict(X_test, batch_size=20)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:',
          np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    return y_pred


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the gesture activity dataset."""
    # custom dataset goes here
    new_data = np.load('11_data.npy')
    X = np.load('final_data.npy')
    X = X[:,0:len(new_data[1,:,1]), :]

    X_train = np.zeros((0,1032,9))
    y_train = np.zeros((0,1))
    act = [3,5,6,11,12,14,15,16,17,19]

    for j in range(0, len(act)):
        X_train = np.concatenate((X_train,
                                  new_data[(act[j] - 1) * 10:(act[j]) * 10,
                                           :, :]), axis = 0)
        X_train = np.concatenate((X_train,
                                  X[(act[j] - 1) * 100:(act[j]) * 100,
                                    :, :]), axis = 0)
        y_train = np.concatenate((y_train,
                                  j * np.ones((len(new_data[(act[j] - 1)
                                                            * 10:(act[j]) * 10,
                                                            :, :]), 1))), axis = 0)
        y_train = np.concatenate((y_train,
                                  j * np.ones((len(X[(act[j] - 1)
                                                     * 100:(act[j]) * 100,
                                                     :, :]), 1))), axis = 0)
    
    y_train = to_categorical(y_train)
    X_test = X_train
    y_test = y_train
    return X_train, y_train, X_test, y_test

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
                        help="Number of iterations used in routing \
                            algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in \
                            each direction.")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be \
                            specified when testing")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data()
    model, eval_model, manipulate_model = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=10,
                                                  routings=args.routings)
    model.summary()
    train(model, X_train, y_train, X_test, y_test)
    test(eval_model, X_test, y_test)
