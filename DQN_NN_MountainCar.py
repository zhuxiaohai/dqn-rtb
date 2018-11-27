# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:41:50 2018

@author: osksti
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class q_estimator:
    """
    This class will initialize and build our model, i.e. the DQN. The DQN
    will be a fully-connected feed forward NN.
    """
    def __init__(self, state_size, action_size, variable_scope):
        self.scope = variable_scope
        self.state_size = state_size
        self.action_size = action_size
        self.input_pl = tf.placeholder(dtype=np.float32, shape=(None, self.state_size),\
                                       name=self.scope+'input_pl')
        self.target_pl = tf.placeholder(dtype=np.float32, shape=(None, self.action_size),\
                                        name=self.scope+'output_pl')
        self.input_layer = tf.layers.dense(self.input_pl, 128, activation=tf.nn.relu,\
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.random_normal,
                                           name=self.scope+'.input_layer')
        self.hidden_layer = tf.layers.dense(self.input_layer, 64, activation=tf.nn.relu,\
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.random_normal,
                                           name=self.scope+'.hidden_layer')
        self.output_layer = tf.layers.dense(self.hidden_layer, self.action_size,\
                                            activation=None,\
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.random_normal,
                                           name=self.scope+'.output_layer')
        self.loss = tf.losses.mean_squared_error(self.target_pl, self.output_layer)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.var_init = tf.global_variables_initializer()
        
    def predict_single(self, sess, state):
        return sess.run(self.output_layer,
                        feed_dict={self.input_pl: np.expand_dims(state, axis=0)})
    
    def predict_batch(self, sess, states):
        return sess.run(self.output_layer, feed_dict={self.input_pl: states})
    
    def train_batch(self, sess, inputs, targets):
        sess.run(self.optimizer, 
                 feed_dict={self.input_pl: inputs, self.target_pl: targets})

class replay_memory:
    """
    This class will define and construct the replay memory.
    """
    def __init__(self, memory_cap, batch_size):
        self.memory_cap = memory_cap
        self.batch_size = batch_size
        self.storage = []
    
    def store_sample(self, sample):            
        if (len(self.storage) == self.memory_cap):
            self.storage.pop(0)
            self.storage.append(sample)
        else:
            self.storage.append(sample)
    
    def get_sample(self):
        if (len(self.storage) <= self.batch_size):
            batch_size = len(self.storage)
        else:
            batch_size = self.batch_size
        
        A = []
        S = np.zeros([batch_size, len(self.storage[0][1])])
        R = np.zeros(batch_size)
        S_prime = np.zeros([batch_size, len(self.storage[0][3])])
        T = []
        
        random_points = []
        counter = 0
        
        while (counter < batch_size):
            index = np.random.randint(0, len(self.storage))
            if (index not in random_points):
                A.append(self.storage[index][0])
                S[counter, :] = self.storage[index][1]
                R[counter] = self.storage[index][2]
                S_prime[counter, :] = self.storage[index][3]
                T.append(self.storage[index][4])
                
                random_points.append(index)
                counter += 1
            else:
                continue
        
        return A, S, R, S_prime, T

class e_greedy_policy:
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay_rate):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.epsilon = self.epsilon_max
        
    def epsilon_update(self, t):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) \
                                            *np.exp(-self.epsilon_decay_rate*t)
    
    
    def action(self, sess, state, q_estimator):
        if (np.random.rand() < self.epsilon):
            return np.random.randint(q_estimator.action_size)
        else:
            action_values = q_estimator.predict_single(sess, state)
            return np.argmax(action_values)
        
        
class agent:
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay_rate, 
                 discount_factor, batch_size, memory_cap,
                 state_size, action_size, sess):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory_cap = memory_cap
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess

        self.reward_episode = 0
        self.reward_list = []
        self.q_estimator = q_estimator(self.state_size, self.action_size, 'q_estimator')
        self.q_target = q_estimator(self.state_size, self.action_size, 'q_target')
        self.e_greedy_policy = e_greedy_policy(self.epsilon_max, self.epsilon_min,
                                               self.epsilon_decay_rate)
        self.replay_memory = replay_memory(self.memory_cap, self.batch_size)
        self.sess.run(self.q_estimator.var_init)
        self.sess.run(self.q_target.var_init)
        
    def action(self, state):
        return self.e_greedy_policy.action(self.sess, state, self.q_estimator)
        
    def q_learning(self):
        action_list, state_matrix, reward_vector,\
        next_state_matrix, termination_list = self.replay_memory.get_sample()
        
        current_q = self.q_estimator.predict_batch(self.sess, state_matrix)
        next_q = self.q_target.predict_batch(self.sess, next_state_matrix)
        
        for i in range(len(action_list)):
            if (termination_list[i] == True):
                current_q[i, action_list[i]] = reward_vector[i]
            else:
                current_q[i, action_list[i]] = reward_vector[i] \
                + self.discount_factor*np.amax(next_q[i, :])
                
        self.q_estimator.train_batch(self.sess, state_matrix, current_q)
        
    def target_network_update(self, polyak_tau=0.95):
        estimator_params = [t for t in tf.trainable_variables() if\
                            t.name.startswith(self.q_estimator.scope)]
        estimator_params = sorted(estimator_params, key=lambda v:v.name)
        target_params = [t for t in tf.trainable_variables() if\
                         t.name.startswith(self.q_target.scope)]
        target_params = sorted(target_params, key=lambda v:v.name)
        update_ops = []
        for e1_v, e2_v in zip(estimator_params, target_params):
            op = e2_v.assign(polyak_tau*e1_v + (1 - polyak_tau)*e2_v)
            update_ops.append(op)
            
        self.sess.run(update_ops)
        
###TESTING---------------------------------------------------------------------        

epsilon_max = 1
epsilon_min = 0.01
epsilon_decay_rate = 0.000025
discount_factor = 0.99
batch_size = 50
memory_cap = 50000
update_frequency = 200
random_n = 25000
episodes_n = 1000

tf.reset_default_graph()

env = gym.make('MountainCar-v0')
env.seed(0)

with tf.Session() as sess:
    agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate, 
                 discount_factor, batch_size, memory_cap,
                 env.observation_space.shape[0], env.action_space.n, sess)
    random_training_counter = 0
    state = env.reset()
    termination = False
    for i in range(random_n):
        action = np.random.randint(env.action_space.n)
        next_state, reward, termination = env.step(action)[:3]
        
        memory_sample = (action, state, reward, next_state, termination)
        agent.replay_memory.store_sample(memory_sample)    
        
        random_training_counter += 1
        print(random_training_counter)
        
        if (termination == True):
            state = env.reset()
        else:
            state = next_state
    
    episode_counter = 1
    global_step_counter = 0
    while (episode_counter < episodes_n):
        if (episode_counter % 10 == 0):
            print('Episode {} of {}'.format(episode_counter, episodes_n))

        state = env.reset()
        termination = False
        while not termination:
            action = agent.action(state)
            next_state, reward, termination = env.step(action)[:3]
            
            agent.reward_episode += reward
            memory_sample = (action, state, reward, next_state, termination)
            agent.replay_memory.store_sample(memory_sample)
            
            global_step_counter += 1
            agent.q_learning()
            if (global_step_counter % update_frequency == 0):
                agent.target_network_update()
            
            agent.e_greedy_policy.epsilon_update(global_step_counter)
            state = next_state
            if (global_step_counter > 100000):
                env.render()
        
        print("Step {},Episode reward: {}, Epsilon: {}"\
              .format(global_step_counter, agent.reward_episode, agent.e_greedy_policy.epsilon))
        agent.reward_list.append(agent.reward_episode)
        agent.reward_episode = 0
        episode_counter += 1
        
sess.close()
env.close()
plt.plot(agent.reward_list)
