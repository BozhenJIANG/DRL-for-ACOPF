'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-04-15 18:57:12
LastEditors: JIANG Bozhen
LastEditTime: 2024-10-20 19:17:10
'''
import numpy as np
import tensorflow as tf
import keras.backend as K
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Reshape, LSTM, Lambda, Flatten
import tensorflow_probability as tfp

class Actor:
    """ Actor Network for the PPO Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.lr = lr
        self.model = self.network()
        self.eps = 0.2
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
        x2 = Dense(512, activation='linear')(x1)
        x3 = -Dense(512, activation='relu')(x1)
        # x = GaussianNoise(1.0)(x)
        #
        out1 = Dense(self.act_dim, activation='softplus')(x2)
        out2 = Dense(self.act_dim, activation='softplus')(x3)
        # out1 = 0.5*(1.0+Dense(self.act_dim, activation='tanh')(x2))
        # out2 = Dense(self.act_dim, activation='softplus')(x2)

        # out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inputs = inp, outputs = [out1,out2])

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(state,verbose=0)
   
    def train(self, states, old_log_probs, actions, advantages):
        """ Actor Training
        """
        with tf.GradientTape() as tape:
            mu, std = self.model(states)
            action_dists = tfp.distributions.Normal(mu, std)
            # log_probs = action_dists.log_prob(actions)
            log_probs = action_dists.log_prob(tf.clip_by_value(actions, clip_value_min=0, clip_value_max=1))
            ratio = tf.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            # print("log_probs: ", log_probs)
            # print("ratio: ", ratio)
            surr2 = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = tf.reduce_mean(-tf.math.minimum(surr1, surr2))   
            # print(surr1)
            # print(tf.math.minimum(surr1, surr2))  
            grads = tape.gradient(actor_loss, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # print(self.model.layers[-1].get_weights())
            
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
