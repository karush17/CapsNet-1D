"""Implements capsule layers."""

from typing import Any, Tuple

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

class Length(layers.Layer):
    """Computes length of vectors."""

    def call(self, inputs: tf.tensor, **kwargs) -> tf.tensor:
        """Returns the length of vector."""
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape: Tuple[Any]) -> Tuple[Any]:
        """Computes the output dimension."""
        return input_shape[:-1]


class Mask(layers.Layer):
    """Implements the mask layer."""

    def call(self, inputs: tf.tensor, **kwargs) -> tf.tensor:
        """Returns the mask on inputs."""
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=tf.shape(x)[1])
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape: Tuple[Any]) -> Tuple[Any]:
        """Computes the output dimension."""
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors: tf.tensor, axis=-1) -> tf.tensor:
    """Implements the squashing activation."""
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm
                                                           + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """Implements the main capsule layer.
    
    Attributes:
        num_capsule: number of capsules.
        dim_capsule: dimension of each capsule.
        num_routing: number of routings.
        kernel_initializer: kernel function initialization.
    """
    def __init__(self, num_capsule: int, dim_capsule: int, num_routing: int = 3,
                 kernel_initializer: str = 'glorot_uniform',
                 **kwargs):
        """Initializes the capsule layer object."""
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape: Tuple[Any]) -> None:
        """Builds the layer."""
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None," \
        "input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(shape=(self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule),
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Implements the capsule layer."""
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]),
                              elems=inputs_tiled)

        inputs_hat_stopped = K.stop_gradient(inputs_hat)        
        b = tf.zeros(shape=(K.shape(inputs_hat)[0],
                            self.num_capsule, self.input_num_capsule))

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(b, dim=1)
            if i == self.num_routing - 1:
                outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            else:
                outputs = squash(K.batch_dot(c, inputs_hat_stopped, [2, 2]))
                b += K.batch_dot(outputs, inputs_hat_stopped, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape: Tuple[Any]) -> Tuple[Any]:
        """Computes the output dimension."""
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs: tf.tensor, dim_capsule: int,
               n_channels: int, kernel_size: int,
               strides: int, padding: str):
    """Implements the primary convolutional layer."""
    output = layers.Conv1D(filters=dim_capsule*n_channels,
                           kernel_size=kernel_size,
                           strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule],
                             name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)
