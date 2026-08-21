'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-04-15 18:54:50
LastEditors: JIANG Bozhen
LastEditTime: 2024-10-25 17:46:26
'''

import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, concatenate, Reshape, LSTM, Lambda, Flatten


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        # self.target_model = self.network()
        # Function to compute Q-value gradients (Actor Optimization)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim))
        action = Input((self.act_dim,))
        x = Dense(512, activation='relu')(state)
        x1 = Dense(1024, activation='relu')(x)
        x2 = concatenate([Flatten()(x1), action])

        x3 = Dense(512, activation='relu')(x2)
        q1 = Dense(1, activation='linear')(x3)

        x4 = Dense(512, activation='relu')(x2)
        q2 = Dense(1, activation='linear')(x4)

        return Model([state, action], [q1,q2])

    def predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.model.predict(inp)

    def train(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        with tf.GradientTape() as tape:
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            current_Q1, current_Q2 = self.model([states, actions],training=True)  # pred for this minibatch
            # print("current_Q1: ",current_Q1)
            # print("critic_Q: ",critic_target)
            # Compute the loss value for this minibatch.
            # loss_value = loss_fn[0](y[:,:3,:], y_pred[0]) + loss_fn[1](y[:,3:6,:], y_pred[1])
            loss_value = tf.math.reduce_mean(tf.keras.losses.MSE(current_Q1, critic_target) + tf.keras.losses.MSE(current_Q2,critic_target))
            # print("loss_value: ",loss_value)
            # print("critic loss: ",tf.math.reduce_mean(loss_value)," ",current_Q1," ",current_Q1," ",critic_target)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            # print(model.trainable_weights)
            
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value

        # return self.model.train_on_batch([states, actions], [critic_target,critic_target])

