'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-07-28 18:04:44
LastEditors: JIANG Bozhen
LastEditTime: 2024-10-29 15:07:36
'''
import numpy as np
import tensorflow as tf
import keras.backend as K
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Reshape, LSTM, Lambda, Flatten

class Actor:
    """ Actor Network for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.lr = lr
        self.model = self.network()
        # self.target_model = self.network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=(self.env_dim,))
        #
        x1 = Dense(1024, activation='linear')(inp)
        # x = GaussianNoise(1.0)(x)
        #
        x2 = Dense(512, activation='relu')(x1)
        # x = GaussianNoise(1.0)(x)
        #
        out1 = Dense(self.act_dim, activation='relu')(x2)
        out2 = Dense(self.act_dim, activation='relu')(x2)
        # out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inputs = inp, outputs = [out1,out2])

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(state,verbose=0)
   
    def train(self, states, actions, advantages):
        """ Actor Training
        """
        with tf.GradientTape() as tape:
            mu,sigma = (
                self.model(states)
            )
            new_action = tf.clip_by_value(tf.random.normal([1], mu, sigma, tf.float32), clip_value_min=0.00001, clip_value_max=1)

            log_loss =  tf.math.reduce_mean(-0.5*tf.math.log(2*3.1415926*(sigma**2+0.1))
                                            -((new_action-mu)**2)/(2*sigma**2+0.1))

            
            actor_loss = tf.math.reduce_mean(log_loss * advantages)

        grads = tape.gradient(actor_loss, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
        return actor_loss

