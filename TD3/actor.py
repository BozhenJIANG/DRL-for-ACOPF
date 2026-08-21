'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-04-15 18:57:12
LastEditors: JIANG Bozhen
LastEditTime: 2024-10-29 15:06:38
'''
import numpy as np
import tensorflow as tf
import keras.backend as K
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Reshape, LSTM, Lambda, Flatten

class Actor:
    """ Actor Network for the TD3 Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau = tau
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
        x1 = Dense(512, activation='linear')(inp)
        # x = GaussianNoise(1.0)(x)

        x2 = Dense(1024, activation='silu')(x1)

        x3 = Dense(512, activation='silu')(x2)
        # x = GaussianNoise(1.0)(x)
        #
        out = Dense(self.act_dim, activation='silu')(x3)
        # out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inputs = inp, outputs = out)

    def predict(self, state):
        """ Action prediction
        """

        return tf.clip_by_value(self.model.predict(state), clip_value_min=0, clip_value_max=1)

    # def target_predict(self, inp):
    #     """ Action prediction (target network)
    #     """
    #     return self.target_model.predict(inp)

    # def transfer_weights(self):
    #     """ Transfer model weights to target model with a factor of Tau
    #     """
    #     W, target_W = self.model.get_weights(), self.target_model.get_weights()
    #     for i in range(len(W)):
    #         target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
    #     self.target_model.set_weights(target_W)

    def train(self, states, critic):
        """ Actor Training
        """
        # print("actions:", actions)
        with tf.GradientTape() as tape:
            new_actions = (
                self.model(states)
            )  
            # Compute the target Q value
            target_Q1, target_Q2 = critic.model([states, new_actions])
            target_Q = tf.math.minimum(target_Q1, target_Q2)
            # print("target_Q: ",target_Q)
            # print("reduce_target_Q: ",-tf.math.reduce_mean(target_Q))
    
            actor_loss = -tf.math.reduce_mean(target_Q) 

            grads = tape.gradient(actor_loss, self.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            # print(model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return actor_loss

    def supervised_train(self, states, actions):

        # print("actions:", actions)
        with tf.GradientTape() as tape:
            new_actions = (
                self.model(states)
            )
                      
            action_loss = tf.math.reduce_mean(tf.keras.losses.MSE(new_actions,tf.convert_to_tensor(actions,dtype=tf.float32)))

            grads = tape.gradient(action_loss, self.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            # print(model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return action_loss
