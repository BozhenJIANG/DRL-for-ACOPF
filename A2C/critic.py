'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-07-28 18:04:44
LastEditors: JIANG Bozhen
LastEditTime: 2024-07-31 22:22:10
'''
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, concatenate, Reshape, LSTM, Lambda, Flatten

class Critic:
    """ Critic for the A2C Algorithm
    """   
    def __init__(self, inp_dim, out_dim, lr):
        self.env_dim = inp_dim
        self.out_dim = out_dim
        self.lr = lr
        self.model = self.network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=(self.env_dim,))
        #
        x1 = Dense(1024, activation='relu')(inp)
        # x = GaussianNoise(1.0)(x)
        #
        x2 = Dense(512, activation='relu')(x1)
        # x = GaussianNoise(1.0)(x)
        #
        out = Dense(1, activation='linear')(x2)
        # out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inputs = inp, outputs = out)
    
    def predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.model.predict(inp)

    def train(self, states, discounted_rewards):
        """ Train the critic network on batch of sampled experience
        """
        with tf.GradientTape() as tape:
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            current_Q =self.model(states,training=True)  # pred for this minibatch
            # print("current_Q1: ",current_Q1)
            # print("critic_Q: ",critic_target)
            # Compute the loss value for this minibatch.
            # loss_value = loss_fn[0](y[:,:3,:], y_pred[0]) + loss_fn[1](y[:,3:6,:], y_pred[1])
            loss_value = tf.math.reduce_mean(tf.keras.losses.MSE(current_Q, discounted_rewards))
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