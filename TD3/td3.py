# '''
# Descripttion: 
# Author: JIANG Bozhen
# version: 
# Date: 2024-04-15 18:57:12
# LastEditors: JIANG Bozhen
# LastEditTime: 2024-10-29 15:06:16
# '''

# import numpy as np
# import os
# import logging

# from tqdm import tqdm
# from .actor import Actor
# from .critic import Critic
# from utils.memory_buffer import MemoryBuffer
# from utils.SettingParametersClass import SettingParameters
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler

# from utils.utilize import *

# # python-Julia接口学习
# from julia.api import Julia
# j = Julia(compiled_modules=False)

# from julia import Main as jl

# class TD3:
#     """ TD3 Main Algorithm
#     """
#     def __init__(self, act_dim, env_dim, gen_num, bus_num, args):

#         """ Initialization
#         """
#         # Environment and TD3 parameters
#         self.case_name = args.case
#         self.act_dim = act_dim
#         self.env_dim = env_dim
#         self.gamma = args.gamma
#         self.lr = args.lr
#         # Create actor and critic networks
#         self.actor = Actor(self.env_dim, act_dim, 0.1*args.lr, args.tau)
#         self.actor_target = Actor(self.env_dim, act_dim, 0.1*args.lr, args.tau)
#         self.critic = Critic(self.env_dim, act_dim, args.lr, args.tau)
#         self.critic_target = Critic(self.env_dim, act_dim, args.lr, args.tau)
#         self.buffer_size = args.buffer_size
#         self.buffer = MemoryBuffer(args.buffer_size)
#         self.tau= args.tau
#         self.algo_name = args.type
#         self.load_weight_episode = args.load_weight_episode

#         self.gen_num = gen_num
#         self.bus_num = bus_num

#         self.basic_batch_size = args.batch_size
#         self.basic_update_times = args.update_times
#         self.freq = args.frequency
#         self.expand_states = args.expand_states

#         self.noise = args.noise 
#         self.nb_episodes = args.nb_episodes+1

#         self.setting = SettingParameters()
        
#         directory = "./"+self.case_name+"/"+ self.algo_name
#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         # 配置日志
#         logging.basicConfig(level=logging.INFO,
#                             format='%(asctime)s %(levelname)s: %(message)s',
#                             datefmt='%Y-%m-%d %H:%M:%S',
#                             filename="./"+self.case_name+"/"+self.algo_name+'/train_log.txt',
#                             filemode='w')
        
#         # 示例：模拟训练过程记录日志
#         logging.info('Start TD3 instantiation')

#         jl.include("./learning.jl")
#         self.jl = jl.Learning
#         if self.case_name == "case24":
#             self.jl._update_para(15,33,24) 
            
#         elif self.case_name == "case118":
#             self.jl._update_para(30,54,118)
        
#         elif self.case_name == "case1354":
#             self.jl._update_para(4231,260,1354)

#         _bus_load = self.jl._get_load(self.case_name)
#         bus_load = np.array(sorted(np.array(_bus_load), key=lambda x: x[0]))
#         self.normal = args.normal
#         self.bus_id,loadp,loadq = creat_dataset(bus_load)
#         self.state_shape = loadp.shape[-1]
#         self.train_dataset,self.test_dataset = train_and_test_data_split(loadp,loadq)
#         self._non_zero_index = np.nonzero(self.train_dataset[0,0,:])

#         if self.normal:
#             scaler_p = StandardScaler()
#             scaler_p.fit((self.train_dataset[:,:24,:]).reshape(-1,self.state_shape))
#             scaler_q = StandardScaler()
#             scaler_q.fit((self.train_dataset[:,24:,:]).reshape(-1,self.state_shape))
#             self.p_mean_ = scaler_p.mean_
#             self.p_var_ = scaler_p.var_
#             self.q_mean_ = scaler_q.mean_
#             self.q_var_ = scaler_q.var_
        
#         if args.load_weight :
#             self.load_weights(args.load_weight_episode)

#     def policy_action(self, s):
#         """ Use the actor_target to predict value
#         """       
#         if self.noise:
#             _action = self.actor.predict(s)
#             return tf.clip_by_value(_action+
#                                     tf.random.truncated_normal(_action.shape,
#                                                                mean=0,
#                                                                stddev=0.1),
#                                     clip_value_min=0,
#                                     clip_value_max=1).numpy()
#         else:
#             return np.clip(self.actor.predict(s),
#                                     a_min=0,
#                                     a_max=1)

#     def bellman(self, rewards, q_values, dones):
#         """ Use the Bellman Equation to compute the critic target
#         """        
#         critic_target = rewards + dones * self.gamma * tf.squeeze(q_values)
#         return critic_target[:,None]

#     def memorize(self, state, action, reward, done, new_state):
#         """ Store experience in memory buffer
#         """
#         self.buffer.memorize(state, action, reward, done, new_state)

#     def sample_batch(self, batch_size):
#         return self.buffer.sample_batch(batch_size)
    
#     def actor_transfer_weights(self):
#         """ Transfer model weights to target model with a factor of Tau
#         """
#         W, target_W = self.actor.model.get_weights(), self.actor_target.model.get_weights()
#         for i in range(len(W)):
#             target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
#         self.actor_target.model.set_weights(target_W)

#     def critic_transfer_weights(self):
#         """ Transfer model weights to target model with a factor of Tau
#         """
#         # print("critic_target.model: ",self.critic_target.model.get_weights()[0][0])
#         # print("critic.model: ",self.critic.model.get_weights()[0][0])
#         W, target_W = self.critic.model.get_weights(), self.critic_target.model.get_weights()
#         for i in range(len(W)):
#             target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
#         self.critic_target.model.set_weights(target_W)
#         # print("critic_target_transfer.model: ",self.critic_target.model.get_weights()[0][0])


#     def update_env(self,timestep,new_env,sample_episode,sample_episode_list,e_):
        
#         if timestep < 23:
#             for _index,_data in enumerate(new_env["bus"]):
#                 _source_id = _data["source_id"][1]
#                 if _source_id in self.bus_id.tolist():

#                     _data["pd"] = self.train_dataset[sample_episode,
#                                                         timestep+1,
#                                                         self.bus_id.tolist().index(_source_id)]
                    
#                     _data["qd"] = self.train_dataset[sample_episode,
#                                                         timestep+24+1,
#                                                         self.bus_id.tolist().index(_source_id)]
                    
#                     new_env["bus"][_index] = _data
#         else:
#             for _index,_data in enumerate(new_env["bus"]):
#                 _source_id = _data["source_id"][1]
#                 if _source_id in self.bus_id.tolist():

#                     _data["pd"] = self.train_dataset[sample_episode_list[e_+1],
#                                                         0,
#                                                         self.bus_id.tolist().index(_source_id)]
#                     _data["qd"] = self.train_dataset[sample_episode_list[e_+1],
#                                                         24,
#                                                         self.bus_id.tolist().index(_source_id)]
#                     new_env["bus"][_index] = _data
        
#         return new_env
    
#     def normal_state(self, env,flows,timestep, normal=False):
#         if normal:
#             state_temp = julia_data_to_python_data(env,flows,timestep,self.expand_states,normal)
#             _temp_data_1 = state_temp[:self.state_shape]
#             _temp_data_2 = state_temp[self.state_shape:2*self.state_shape]
#             state_temp[:self.state_shape] = (_temp_data_1-self.p_mean_)/(self.p_var_+0.0001)
#             state_temp[self.state_shape:2*self.state_shape] = (_temp_data_2-self.q_mean_)/(self.q_var_+0.0001)
#             return state_temp
#         else:
#             return julia_data_to_python_data(env,flows,timestep,self.expand_states)
    
#     def train(self, summary_writer):
#         reward_list = []
#         r_true_list = []
#         # self.load_weights(episode=599)

#         #First, gather experience
#         tqdm_e = tqdm(range(self.nb_episodes), desc='Score', leave=True, unit=" episodes") 
#         sample_episode_list = np.random.randint(800,size=self.nb_episodes)
#         #开始训练
#         logging.info('The model starts training')       
#         for e_,episode in enumerate(tqdm_e):
#             logging.info(f'Episode {episode} is in progress...')
            
#             not_legal_times = 0
#             # sample_episode = np.random.randint(8000,size=1)
#             sample_episode = sample_episode_list[e_]
#             # 随机初始化
#             # create env 
#             env, flows = self.jl.create_env(self.case_name,
#                                             self.bus_id,
#                                             self.train_dataset[sample_episode,0,:],
#                                             self.train_dataset[sample_episode,0+24,:]
#                                             )
#             PV_bus_set = get_PV_bus_set(env)

#             _info = "True"

#             _env = env
#             agent_reward = 0.0
#             solver_reward = 0.0

#             r_true = 0
#             r_true_flag = 0

#             _critic_loss = 0
#             _actor_loss = 0

#             for timestep in range(24):
#                 # print(timestep+1)
#                 logging.info(f'Timestep {timestep} is in progress...')
#                 # 对s进行标准化

#                 state = self.normal_state(env,
#                                           flows, 
#                                           timestep, 
#                                           normal=self.normal)           
                     
#                 a = self.policy_action(state.reshape((1,-1)))

#                 action = inverse_a2action(np.reshape(a,[-1]),
#                                            env,
#                                            self.gen_num,
#                                            PV_bus_set)
#                 # print(action)

#                 # if timestep == 0:
#                 #     new_env, r, done, info = jl.actor_solve_and_step(env,
#                 #                                 self.train_dataset[sample_episode,timestep,:],
#                 #                                 self.train_dataset[sample_episode,timestep+24,:],
#                 #                                 action,
#                 #                                 _env,"true") 
#                 # else:
#                 # print(a)
#                 # print("Agent Action:") 
#                 # print([round(_temp_action,3) for _temp_action in np.reshape(a,[-1])])    
#                 if sample_episode+1 < self.nb_episodes:
#                     if timestep+1 <24 :
#                         (new_env,new_flows), (reward_without_Q, r), done, info = self.jl.actor_solve_and_step(env,
#                                                     self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                     self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                     action,
#                                                     _env) 
#                         # (new_env,new_flows), a, (reward_without_Q, r), done, info = self.jl.opf_solve_and_step(env,
#                         #                                                 self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                         #                                                 self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                         #                                                 _env,"true")                           
#                     else:
#                         break
#                 else:
#                     # Edge case: near end of training, just use current env
#                     if timestep+1 <24 :
#                         (new_env,new_flows), (reward_without_Q, r), done, info = self.jl.actor_solve_and_step(env,
#                                                     self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                     self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                     action,
#                                                     _env)
#                     else:
#                         break
#                 # print(info)
#                 # print("!!!!!!!!!!")
#                 # print(self.critic_target.predict([state.reshape((1,-1)),np.reshape(a,[1,-1])]))
#                 # print("!!!!!!!!!!")
#                 # print(action)
#                 # print("q_load:")
#                 # print(self.train_dataset[sample_episode,timestep,:])
#                 # print("p_load:")
#                 # print(self.train_dataset[sample_episode,timestep+24,:])
#                 # break
#                 # if timestep == 0:                
#                 #     print("Agent action: ",tf.squeeze(action).numpy())
#                 #     print("Agentr reward: ", r)
#                 #     print("Agent done: ", done)

#             #     break
#             # break
#                 new_env = self.update_env(timestep,
#                                           new_env,
#                                           sample_episode,
#                                           sample_episode_list,
#                                           e_)

#                 #存储智能体的动作,可能有好有坏，都存入
#                 r_agent = np.array(r)
#                 r_agent_without_Q = np.array(reward_without_Q)
#                 a = np.array(a).reshape(-1)
#                 if timestep < 23:
#                     state_ = self.normal_state(new_env,
#                                                 new_flows,
#                                                 timestep+1,
#                                                 normal=self.normal)
#                 else:
#                     state_ = self.normal_state(new_env,
#                                                 new_flows,
#                                                 25,
#                                                 normal=self.normal)

#                 if info == "True" :
#                     done_bool = 1
#                     #存储临时环境
#                     nextenv = new_env
#                 else:
#                     done_bool = 0

#                 # if np.random.random() < 0.01:
#                 #     self.memorize(state, 
#                 #                   a, 
#                 #                   r_agent, 
#                 #                   done_bool, 
#                 #                   state_)

                               
#                 if info == "True" :
#                     logging.info(f'Timestep {timestep} : Agent solution feasible')
#                     #记录Agent真实的回报
#                     if r_true_flag == 0:
#                         r_true += r 

#                     # 如果该动作并不是很好，给出更好动作
#                     if r < 1:
#                         logging.info(f'Timestep {timestep} : Agent solution is not good')
#                         # using IPOPT to give better solution
#                         ##############################################
#                         #  由于解的不唯一性，SOLVER给出的结果不应该存入进去，应该把PF算出的结果存入进去！！！！！！！！！
#                         ##############################################
#                         # print("Solver Action:") 
#                         (new_env,new_flows), a, (reward_without_Q, r), done, info = self.jl.opf_solve_and_step(env,
#                                                                         self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                                         self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                                         _env)
#                         # print(a)
#                         # check the Solver solution whether fesible
#                         action = inverse_a2action(a,
#                                                    env,
#                                                    self.gen_num,
#                                                    PV_bus_set)
#                         # print("Solver Action:")  
#                         # print([round(_temp_action,3) for _temp_action in np.reshape(a,[-1])])                     
#                         (_new_env,_new_flows), (_reward_without_Q, _r), _done, _info = self.jl.actor_solve_and_step(env,
#                                                                             self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                                             self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                                             action,
#                                                                             _env)   
#                         # break
#                         if _info == "False":  
#                             print("Solver solution is not feasible") 
#                             break
                            
#                         else:
#                             new_env = self.update_env(timestep,
#                                                       new_env,
#                                                       sample_episode,
#                                                       sample_episode_list,
#                                                       e_)

#                         #存储更好的动作
#                         if timestep < 23:
#                             state_ = self.normal_state(new_env,
#                                                        new_flows,
#                                                        timestep+1,
#                                                        normal=self.normal)
#                         else:
#                             state_ = self.normal_state(new_env,
#                                                        new_flows,
#                                                        25,
#                                                        normal=self.normal)
                            
#                         r = np.array(_r)
#                         a = np.array(a).reshape(-1)

#                         if timestep+1 == 24:
#                             done_bool = 0
#                         else:
#                             done_bool = 1. - float(done)

#                         # if np.random.random() < 0.1:
#                         #     self.memorize(state, 
#                         #                 a, 
#                         #                 r, 
#                         #                 done_bool, 
#                         #                 state_)
#                         solver_reward += _r
#                     else:
#                         logging.info(f'Timestep {timestep} : Agent solution is good')
#                         solver_reward += r
                     
#                     # #更新环境 ：以智能体的状态更新环境
#                     # env = nextenv

#                     #更新环境 ：以智能体的状态更新环境
#                     env = new_env

#                 else:
#                     logging.info(f'Timestep {timestep} : Agent solution is not feasible')
#                     # break
#                     # 记录第一次出错的时间步
#                     if r_true_flag == 0:
#                         first_wrong_timestamp = timestep
#                         r_true_flag = 1
#                         r_true_list.append(r_true)

#                     # using IPOPT to give optimal solution  
#                     # print("Solver Action:")    
#                     # print(episode+1," ",timestep+1)                 
#                     (new_env,new_flows), a, (reward_without_Q, r), done, info = self.jl.opf_solve_and_step(env,
#                                                                       self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                                       self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                                       new_env)
#                     # print(a)
#                     # break

#                     # check the Solver solution whether fesible
#                     action = inverse_a2action(a,
#                                                env,
#                                                self.gen_num,
#                                                PV_bus_set)

#                     # if timestep == 0:
#                     #     print("Solver action: ",tf.squeeze(action).numpy())
#                     #     print("Solver reward: ", r)
#                     #     print("Solver done: ", done)
#                     # print(action)
#                     # print("Excute Solver Action:") 
#                     (_new_env,_new_flows), (_reward_without_Q, _r), _done, _info = self.jl.actor_solve_and_step(env,
#                                                                          self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                                          self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                                          action,
#                                                                          _env)
                    

#                     # print(self.train_dataset[sample_episode,timestep,:])   
#                     # break
#                     if _info == "False":  
#                         print(episode," ",timestep," Solver solution is not feasible")
#                         _new_env, (_reward_without_Q, _r), _done, _info = self.jl.actor_solve_and_step(env,
#                                                                                   self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                                                   self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                                                   action,
#                                                                                   _env)
#                         _new_env, a, (_reward_without_Q, _r), done, info = self.jl.opf_solve_and_step(env,
#                                                                                 self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
#                                                                                 self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
#                                                                                 _env)
#                         break
                        
#                     else:
#                         new_env = self.update_env(timestep,
#                                                   new_env,
#                                                   sample_episode,
#                                                   sample_episode_list,
#                                                   e_)
#                         solver_reward += _r
#                         not_legal_times += 1

#                         #存储正确动作
#                         if timestep < 23:
#                             state_ = self.normal_state(new_env,
#                                                        new_flows,
#                                                        timestep+1,
#                                                        normal=self.normal)
#                         else:
#                             state_ = self.normal_state(new_env,
#                                                        new_flows,
#                                                        25,
#                                                        normal=self.normal)
                            
#                         env = new_env
#                         r = np.array(_r)
#                         a = np.array(a).reshape(-1)

#                         if timestep+1 == 24:
#                             done_bool = 0
#                         else:
#                             done_bool = 1. - float(done)

#                         self.memorize(state, 
#                                       a, 
#                                       r, 
#                                       done_bool, 
#                                       state_)

#                 # if _info == "False":   
#                 #     break  
#                 # break
#                 #存储正确动作

#                 agent_reward += r_agent  #记录当前EP的总reward
#                 # agent_reward += r_agent_without_Q  #记录当前EP的总reward
#                 if self.buffer.count > 4400:

#                     logging.info(f'Timestep {timestep} : Agent critic training')
#                     if (timestep+1) % (2*self.freq) == 0 :
#                         # 每交互1次，抽取batch_size，进行update_times次critic的训练                                  
#                         k = 1 + (self.buffer.count) / self.buffer_size

#                         batch_size   = int(k * self.basic_batch_size)
#                         update_times = int(k * self.basic_update_times)

#                         for index_ in range(update_times):

#                             # Sample experience from buffer
#                             states, actions, rewards, dones, new_states, _ = self.sample_batch(batch_size)

#                             # Predict target q-values using target networks
#                             # print(new_states)
#                             next_action = self.actor_target.predict(new_states)
#                             target_Q1, target_Q2 = self.critic_target.predict([new_states,
#                                                                             next_action])
#                             target_Q = tf.math.minimum(target_Q1, 
#                                                     target_Q2)

#                             # Compute critic target
#                             critic_target_Q = self.bellman(rewards, 
#                                                         target_Q, 
#                                                         dones)

#                             # Train critic
#                             critic_loss = self.critic.train(states, 
#                                                             actions, 
#                                                             critic_target_Q)
#                             _critic_loss += critic_loss
                            
#                             # print("critic.model: ",self.critic.model([states,actions])[0][0],self.critic.model([states,actions])[1][0])
#                             # print("Actual: ",critic_target[0])
#                             # print(self.critic_target.model.get_weights()[0][0])

#                     logging.info(f'Timestep {timestep} : Agent actor training')
#                     # critic每训练4次，才对actor进行训练    
#                     if (timestep+1) % (4*self.freq) == 0 :

#                         # Q-Value Gradients under Current Policy
#                         actor_loss= self.actor.train(states, 
#                                                     self.critic)

#                         # print(self.critic_target.model.get_weights()[0][0])
#                         _actor_loss += actor_loss

#                         if (episode+1) % (4) == 0 :
#                             actor_loss= self.actor.supervised_train(states,
#                                                                 actions)

#                         self.critic_transfer_weights()
#                         self.actor_transfer_weights()

#                         # if _action_loss < 0.01:
#                         #     print(actions)
#                         #     print(self.actor.model.predict(states))
#                         # if float(_actor_loss)/5 > 24:
#                         #     print("target_Q: ", target_Q)

#                     # 使用 TensorBoard 记录数据
#                     with summary_writer.as_default():
#                         tf.summary.scalar('agent_reward', float(agent_reward), step=episode)
#                         tf.summary.scalar('solver_reward', float(solver_reward), step=episode)
#                         tf.summary.scalar('not_legal_times', int(not_legal_times), step=episode)
#                         tf.summary.scalar('_actor_loss', float(_actor_loss), step=episode)
#                         tf.summary.scalar('_critic_loss', float(_critic_loss), step=episode)
#                         tf.summary.scalar('First Wrong Timestamp', float(first_wrong_timestamp), step=episode)


#             # if _info == "False":   
#             #     break  
#             # break
#             if not_legal_times == 0:
#                 first_wrong_timestamp = 24
#             if self.buffer.count > 4500:
#                 logging.info(f'Episode {episode} : agent_reward {str(round(agent_reward, 2))}, solver_reward {str(round(solver_reward, 2))},not_legal_times {not_legal_times},_actor_loss  {str(round(float(_actor_loss)/3, 2))},_critic_loss {str(round(float(_critic_loss)/6/update_times, 2))}')
#                 reward_list.append([agent_reward,solver_reward,not_legal_times,float(_actor_loss)/3,float(_critic_loss)/6/update_times])
#                 tqdm_e.set_description("Score: " + str(round(agent_reward, 5))+" "+
#                                     "Solver Score: " + str(round(solver_reward, 5))+" "+
#                                     "First Wrong Timestamp: " +str(first_wrong_timestamp) + " "+
#                                     "True Score: " + str(r_true)+" "+
#                                     "Agent Fail Times: " + str(not_legal_times)+" "+
#                                     "Agent Actor Loss: " + str(round(float(_actor_loss)/3, 5))+" "+
#                                     "Agent Critic Loss: " + str(round(float(_critic_loss)/6/update_times, 5)))
#                 tqdm_e.refresh()  
                
#                 if (self.load_weight_episode+episode+1) % 200 == 0: 
#                     # Periodic checkpoint for crash recovery
#                     _rl_path = "./"+self.case_name+"/"+self.algo_name+"/reward_list_"+\
#                                str(self.act_dim)+"_"+str(self.env_dim)+"_"+\
#                                str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+\
#                                str(int(self.expand_states))+"_"+str(int(self.noise))+"_"+\
#                                str(int(self.normal))+".npy"
#                     _trl_path = "./"+self.case_name+"/"+self.algo_name+"/true_reward_list_"+\
#                                 str(self.act_dim)+"_"+str(self.env_dim)+"_"+\
#                                 str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+\
#                                 str(int(self.expand_states))+"_"+str(int(self.noise))+"_"+\
#                                 str(int(self.normal))+".npy"
#                     if os.path.exists(_rl_path):
#                         _old_rl = np.load(_rl_path, allow_pickle=True)
#                         if len(reward_list) > 0:
#                             reward_list = np.vstack((_old_rl, np.array(reward_list)))
#                         else:
#                             reward_list = _old_rl
#                         if len(r_true_list) > 0 and os.path.exists(_trl_path):
#                             _old_trl = np.load(_trl_path, allow_pickle=True)
#                             r_true_list = np.vstack((_old_trl, np.array(r_true_list).reshape(-1,1)))
#                     np.save(_rl_path, np.array(reward_list))
#                     if len(r_true_list) > 0:
#                         np.save(_trl_path, np.array(r_true_list).reshape(-1,1))
#                     reward_list = []
#                     r_true_list = []
#                     self.save_weights(self.load_weight_episode+episode+1)
                    
#             logging.info('Training finish!')
#             # '''
#             # only for test
#             # '''

#             # tqdm_e.set_description()
#             # tqdm_e.refresh()  
    
#         # --- Save final results at end of training ---
#         out_dir = "./"+self.case_name+"/"+self.algo_name
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
#         suffix = (str(self.act_dim)+"_"+str(self.env_dim)+"_"+
#                   str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+
#                   str(int(self.expand_states))+"_"+str(int(self.noise))+"_"+
#                   str(int(self.normal)))
        
#         # Merge with previously saved reward_list (periodic checkpoints or resume)
#         rl_path = out_dir+"/reward_list_"+suffix+".npy"
#         trl_path = out_dir+"/true_reward_list_"+suffix+".npy"
#         if os.path.exists(rl_path):
#             old_rl = np.load(rl_path, allow_pickle=True)
#             if len(reward_list) > 0:
#                 reward_list = np.vstack((old_rl, np.array(reward_list)))
#             else:
#                 reward_list = old_rl
#         np.save(rl_path, np.array(reward_list))
#         if len(r_true_list) > 0:
#             r_true_list_arr = np.array(r_true_list).reshape(-1,1)
#             if os.path.exists(trl_path):
#                 old_trl = np.load(trl_path, allow_pickle=True)
#                 r_true_list_arr = np.vstack((old_trl, r_true_list_arr))
#             np.save(trl_path, r_true_list_arr)
#         self.save_weights(self.load_weight_episode + self.nb_episodes)
#         # Clean up intermediate checkpoints (keep only final)
#         import glob as _glob
#         _final_ep = str(self.load_weight_episode + self.nb_episodes)
#         for _f in _glob.glob(out_dir+"/actor_*.h5") + _glob.glob(out_dir+"/critic_*.h5"):
#             if _final_ep + "_" not in _f:
#                 os.remove(_f)
#         logging.info(f'Final weights and rewards saved to {out_dir}')
#         print(f'[OK] Saved: {rl_path} ({len(reward_list)} episodes)')
#         print(f'[OK] Saved: {out_dir}/actor_*.h5, critic_*.h5')

#         return reward_list

#     def save_weights(self,episode=100):        
#         self.actor.model.save_weights("./"+self.case_name+
#                                       "/"+self.algo_name+
#                                       "/actor_"+str(episode)+"_"+
#                                       str(self.act_dim)+"_"+
#                                       str(self.env_dim)+"_"+
#                                       str(self.gamma)[2:]+"_"+
#                                       str(self.lr)[2:]+"_"+
#                                       str(int(self.expand_states))+"_"+
#                                       str(int(self.noise))+"_"
#                                       +str(int(self.normal))+".h5")
#         self.critic.model.save_weights("./"+self.case_name+
#                                        "/"+self.algo_name+
#                                        "/critic_"+str(episode)+"_"+
#                                       str(self.act_dim)+"_"+
#                                       str(self.env_dim)+"_"+
#                                       str(self.gamma)[2:]+"_"+
#                                       str(self.lr)[2:]+"_"+
#                                       str(int(self.expand_states))+"_"+
#                                       str(int(self.noise))+"_"
#                                       +str(int(self.normal))+".h5")

#     def load_weights(self,episode=100):
#         self.actor.model.load_weights("./"+self.case_name+
#                                       "/"+self.algo_name+
#                                       "/actor_"+str(episode)+"_"+
#                                       str(self.act_dim)+"_"+
#                                       str(self.env_dim)+"_"+
#                                       str(self.gamma)[2:]+"_"+
#                                       str(self.lr)[2:]+"_"+
#                                       str(int(self.expand_states))+"_"+
#                                       str(int(self.noise))+"_"+
#                                       str(int(self.normal))+".h5")
        
#         _W =self.actor.model.get_weights()
#         self.actor_target.model.set_weights(_W)

#         self.critic.model.load_weights("./"+self.case_name+
#                                        "/"+self.algo_name+
#                                        "/critic_"+str(episode)+"_"+
#                                       str(self.act_dim)+"_"+
#                                       str(self.env_dim)+"_"+
#                                       str(self.gamma)[2:]+"_"+
#                                       str(self.lr)[2:]+"_"+
#                                       str(int(self.expand_states))+"_"+
#                                       str(int(self.noise))+"_"+
#                                       str(int(self.normal))+".h5")
        
#         _W =self.critic.model.get_weights()
#         self.critic_target.model.set_weights(_W)

#         # One-time alignment of the reward history with the resumed checkpoint:
#         # a crashed segment may have flushed rows beyond `episode` that are
#         # about to be re-trained. Truncate so that later merges (which append
#         # without truncating) stay episode-aligned.
#         _suffix = (str(self.act_dim)+"_"+str(self.env_dim)+"_"+
#                    str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+
#                    str(int(self.expand_states))+"_"+str(int(self.noise))+"_"+
#                    str(int(self.normal)))
#         _rl_path = "./"+self.case_name+"/"+self.algo_name+"/reward_list_"+_suffix+".npy"
#         _trl_path = "./"+self.case_name+"/"+self.algo_name+"/true_reward_list_"+_suffix+".npy"
#         if os.path.exists(_rl_path):
#             _old_rl = np.load(_rl_path, allow_pickle=True)
#             if len(_old_rl) > episode:
#                 np.save(_rl_path, _old_rl[:episode])
#         if os.path.exists(_trl_path):
#             _old_trl = np.load(_trl_path, allow_pickle=True)
#             if len(_old_trl) > episode:
#                 np.save(_trl_path, _old_trl[:episode])


'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-04-15 18:57:12
LastEditors: JIANG Bozhen
LastEditTime: 2024-10-30 09:52:09
'''

import numpy as np
import os
import logging

from tqdm import tqdm
from .actor import Actor
from .critic import Critic
from utils.memory_buffer import MemoryBuffer
from utils.SettingParametersClass import SettingParameters
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from utils.utilize import *

# python-Julia接口学习
from julia.api import Julia
j = Julia(compiled_modules=False)

from julia import Main as jl

class TD3:
    """ TD3 Main Algorithm
    """
    def __init__(self, act_dim, env_dim, gen_num, bus_num, args):

        """ Initialization
        """
        # Environment and TD3 parameters
        self.case_name = args.case
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = args.gamma
        self.lr = args.lr
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, 0.1*args.lr, args.tau)
        self.actor_target = Actor(self.env_dim, act_dim, 0.1*args.lr, args.tau)
        self.critic = Critic(self.env_dim, act_dim, args.lr, args.tau)
        self.critic_target = Critic(self.env_dim, act_dim, args.lr, args.tau)
        self.buffer_size = args.buffer_size
        self.buffer = MemoryBuffer(args.buffer_size)
        self.tau= args.tau
        self.algo_name = args.type
        self.load_weight_episode = args.load_weight_episode

        self.gen_num = gen_num
        self.bus_num = bus_num

        self.basic_batch_size = args.batch_size
        self.basic_update_times = args.update_times
        self.freq = args.frequency
        self.expand_states = args.expand_states

        self.noise = args.noise 
        self.nb_episodes = args.nb_episodes+1

        self.setting = SettingParameters()
        
        directory = "./"+self.case_name+"/"+ self.algo_name
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 配置日志
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename="./"+self.case_name+"/"+self.algo_name+'/train_log.txt',
                            filemode='w')
        
        # 示例：模拟训练过程记录日志
        logging.info('Start TD3 instantiation')

        jl.include("./learning.jl")
        self.jl = jl.Learning
        if self.case_name == "case24":
            self.jl._update_para(15,33,24) 
            
        elif self.case_name == "case118":
            self.jl._update_para(30,54,118)
        
        elif self.case_name == "case1354":
            self.jl._update_para(4231,260,1354)

        _bus_load = self.jl._get_load(self.case_name)
        bus_load = np.array(sorted(np.array(_bus_load), key=lambda x: x[0]))
        self.normal = args.normal
        self.bus_id,loadp,loadq = creat_dataset(bus_load)
        self.state_shape = loadp.shape[-1]
        self.train_dataset,self.test_dataset = train_and_test_data_split(loadp,loadq)
        self._non_zero_index = np.nonzero(self.train_dataset[0,0,:])

        if self.normal:
            scaler_p = StandardScaler()
            scaler_p.fit((self.train_dataset[:,:24,:]).reshape(-1,self.state_shape))
            scaler_q = StandardScaler()
            scaler_q.fit((self.train_dataset[:,24:,:]).reshape(-1,self.state_shape))
            self.p_mean_ = scaler_p.mean_
            self.p_var_ = scaler_p.var_
            self.q_mean_ = scaler_q.mean_
            self.q_var_ = scaler_q.var_
        
        if args.load_weight :
            self.load_weights(args.load_weight_episode)

    def policy_action(self, s):
        """ Use the actor_target to predict value
        """       
        if self.noise:
            _action = self.actor.predict(s)
            return tf.clip_by_value(_action+
                                    tf.random.truncated_normal(_action.shape,
                                                               mean=0,
                                                               stddev=0.1),
                                    clip_value_min=0,
                                    clip_value_max=1).numpy()
        else:
            return self.actor.predict(s)

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """        
        critic_target = rewards + dones * self.gamma * tf.squeeze(q_values)
        return critic_target[:,None]

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)
    
    def actor_transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.actor.model.get_weights(), self.actor_target.model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.actor_target.model.set_weights(target_W)

    def critic_transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        # print("critic_target.model: ",self.critic_target.model.get_weights()[0][0])
        # print("critic.model: ",self.critic.model.get_weights()[0][0])
        W, target_W = self.critic.model.get_weights(), self.critic_target.model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.critic_target.model.set_weights(target_W)
        # print("critic_target_transfer.model: ",self.critic_target.model.get_weights()[0][0])


    def update_env(self,timestep,new_env,sample_episode,sample_episode_list,e_):
        
        if timestep < 23:
            for _index,_data in enumerate(new_env["bus"]):
                _source_id = _data["source_id"][1]
                if _source_id in self.bus_id.tolist():

                    _data["pd"] = self.train_dataset[sample_episode,
                                                        timestep+1,
                                                        self.bus_id.tolist().index(_source_id)]
                    
                    _data["qd"] = self.train_dataset[sample_episode,
                                                        timestep+24+1,
                                                        self.bus_id.tolist().index(_source_id)]
                    
                    new_env["bus"][_index] = _data
        else:
            for _index,_data in enumerate(new_env["bus"]):
                _source_id = _data["source_id"][1]
                if _source_id in self.bus_id.tolist():

                    _data["pd"] = self.train_dataset[sample_episode_list[e_+1],
                                                        0,
                                                        self.bus_id.tolist().index(_source_id)]
                    _data["qd"] = self.train_dataset[sample_episode_list[e_+1],
                                                        24,
                                                        self.bus_id.tolist().index(_source_id)]
                    new_env["bus"][_index] = _data
        
        return new_env
    
    def normal_state(self, env,flows,timestep, normal=False):
        if normal:
            state_temp = julia_data_to_python_data(env,flows,timestep,self.expand_states,normal)
            _temp_data_1 = state_temp[:self.state_shape]
            _temp_data_2 = state_temp[self.state_shape:2*self.state_shape]
            state_temp[:self.state_shape] = (_temp_data_1-self.p_mean_)/(self.p_var_+0.0001)
            state_temp[self.state_shape:2*self.state_shape] = (_temp_data_2-self.q_mean_)/(self.q_var_+0.0001)
            return state_temp
        else:
            return julia_data_to_python_data(env,flows,timestep,self.expand_states)
    
    def train(self, summary_writer):
        reward_list = []
        r_true_list = []
        # self.load_weights(episode=599)

        #First, gather experience
        tqdm_e = tqdm(range(self.nb_episodes), desc='Score', leave=True, unit=" episodes") 
        sample_episode_list = np.random.randint(800,size=self.nb_episodes)
        #开始训练
        logging.info('The model starts training')       
        for e_,episode in enumerate(tqdm_e):
            logging.info(f'Episode {episode} is in progress...')
            
            not_legal_times = 0
            # sample_episode = np.random.randint(8000,size=1)
            sample_episode = sample_episode_list[e_]
            # 随机初始化
            # create env 
            env, flows = self.jl.create_env(self.case_name,
                                            self.bus_id,
                                            self.train_dataset[sample_episode,0,:],
                                            self.train_dataset[sample_episode,0+24,:]
                                            )
            PV_bus_set = get_PV_bus_set(env)

            _info = "True"

            _env = env
            agent_reward = 0.0
            solver_reward = 0.0

            r_true = 0
            r_true_flag = 0

            _critic_loss = 0
            _actor_loss = 0
            first_wrong_timestamp = 24  # initialized for TensorBoard logging

            for timestep in range(24):
                # print(timestep+1)
                logging.info(f'Timestep {timestep} is in progress...')
                # 对s进行标准化

                state = self.normal_state(env,
                                          flows, 
                                          timestep, 
                                          normal=self.normal)           
                     
                a = self.policy_action(state.reshape((1,-1)))

                action = inverse_a2action(np.reshape(a,[-1]),
                                           env,
                                           self.gen_num,
                                           PV_bus_set)

                # if timestep == 0:
                #     new_env, r, done, info = jl.actor_solve_and_step(env,
                #                                 self.train_dataset[sample_episode,timestep,:],
                #                                 self.train_dataset[sample_episode,timestep+24,:],
                #                                 action,
                #                                 _env,"true") 
                # else:
                # print(a)
                # print("Agent Action:") 
                # print([round(_temp_action,3) for _temp_action in np.reshape(a,[-1])])    
                if sample_episode+1 < self.nb_episodes:
                    if timestep+1 <24 :
                        (new_env,new_flows), (reward_without_Q, r), done, info = self.jl.actor_solve_and_step(env,
                                                    self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                    self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                    action,
                                                    _env)
                                                    # ,"true") 
                    else:
                        break
                else:
                    # Edge case: near end of training, just use current env
                    if timestep+1 <24 :
                        (new_env,new_flows), (reward_without_Q, r), done, info = self.jl.actor_solve_and_step(env,
                                                    self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                    self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                    action,
                                                    _env)
                    else:
                        break
                # print(info)
                # print("!!!!!!!!!!")
                # print(self.critic_target.predict([state.reshape((1,-1)),np.reshape(a,[1,-1])]))
                # print("!!!!!!!!!!")
                # print(action)
                # print("q_load:")
                # print(self.train_dataset[sample_episode,timestep,:])
                # print("p_load:")
                # print(self.train_dataset[sample_episode,timestep+24,:])
                # break
                # if timestep == 0:                
                #     print("Agent action: ",tf.squeeze(action).numpy())
                #     print("Agentr reward: ", r)
                #     print("Agent done: ", done)


                new_env = self.update_env(timestep,
                                          new_env,
                                          sample_episode,
                                          sample_episode_list,
                                          e_)

                #存储智能体的动作,可能有好有坏，都存入
                r_agent = np.array(r)
                r_agent_without_Q = np.array(reward_without_Q)
                a = np.array(a).reshape(-1)
                if timestep < 23:
                    state_ = self.normal_state(new_env,
                                                new_flows,
                                                timestep+1,
                                                normal=self.normal)
                else:
                    state_ = self.normal_state(new_env,
                                                new_flows,
                                                25,
                                                normal=self.normal)

                if info == "True" :
                    done_bool = 1
                    #存储临时环境
                    nextenv = new_env
                else:
                    done_bool = 0
                if np.random.random() < 0.5:
                    self.memorize(state, 
                                  a, 
                                  r_agent, 
                                  done_bool, 
                                  state_)

                               
                if info == "True" :
                    logging.info(f'Timestep {timestep} : Agent solution feasible')
                    #记录Agent真实的回报
                    if r_true_flag == 0:
                        r_true += r 

                    # 如果该动作并不是很好，给出更好动作
                    if r < 1:
                        logging.info(f'Timestep {timestep} : Agent solution is not good')
                        # using IPOPT to give better solution
                        ##############################################
                        #  由于解的不唯一性，SOLVER给出的结果不应该存入进去，应该把PF算出的结果存入进去！！！！！！！！！
                        ##############################################
                        # print("Solver Action:") 
                        (new_env,new_flows), a, (reward_without_Q, r), done, info = self.jl.opf_solve_and_step(env,
                                                                        self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                        self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                        _env)
                                                                        # ,"true")
                        # print(a)
                        # check the Solver solution whether fesible
                        action = inverse_a2action(a,
                                                   env,
                                                   self.gen_num,
                                                   PV_bus_set)
                        # print("Solver Action:")  
                        # print([round(_temp_action,3) for _temp_action in np.reshape(a,[-1])])                     
                        (_new_env,_new_flows), (_reward_without_Q, _r), _done, _info = self.jl.actor_solve_and_step(env,
                                                                            self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                            self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                            action,
                                                                            _env)   
                        # break
                        if _info == "False":  
                            print("Solver solution is not feasible") 
                            break
                            
                        else:
                            new_env = self.update_env(timestep,
                                                      new_env,
                                                      sample_episode,
                                                      sample_episode_list,
                                                      e_)

                        #存储更好的动作
                        if timestep < 23:
                            state_ = self.normal_state(new_env,
                                                       new_flows,
                                                       timestep+1,
                                                       normal=self.normal)
                        else:
                            state_ = self.normal_state(new_env,
                                                       new_flows,
                                                       25,
                                                       normal=self.normal)
                            
                        r = np.array(_r)
                        a = np.array(a).reshape(-1)

                        if timestep+1 == 24:
                            done_bool = 0
                        else:
                            done_bool = 1. - float(done)

                        if np.random.random() < 0.1:
                            self.memorize(state, 
                                        a, 
                                        r, 
                                        done_bool, 
                                        state_)
                        solver_reward += _r
                    else:
                        logging.info(f'Timestep {timestep} : Agent solution is good')
                        solver_reward += r
                     
                    # #更新环境 ：以智能体的状态更新环境
                    # env = nextenv

                    #更新环境 ：以智能体的状态更新环境
                    env = new_env

                else:
                    logging.info(f'Timestep {timestep} : Agent solution is not feasible')
                    # break
                    # 记录第一次出错的时间步
                    if r_true_flag == 0:
                        first_wrong_timestamp = timestep
                        r_true_flag = 1
                        r_true_list.append(r_true)

                    # using IPOPT to give optimal solution  
                    # print("Solver Action:")    
                    # print(episode+1," ",timestep+1)                 
                    (new_env,new_flows), a, (reward_without_Q, r), done, info = self.jl.opf_solve_and_step(env,
                                                                      self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                      self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                      new_env)
                    # print(a)
                    # break

                    # check the Solver solution whether fesible
                    action = inverse_a2action(a,
                                               env,
                                               self.gen_num,
                                               PV_bus_set)

                    # if timestep == 0:
                    #     print("Solver action: ",tf.squeeze(action).numpy())
                    #     print("Solver reward: ", r)
                    #     print("Solver done: ", done)
                    # print(action)
                    # print("Excute Solver Action:") 
                    (_new_env,_new_flows), (_reward_without_Q, _r), _done, _info = self.jl.actor_solve_and_step(env,
                                                                         self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                         self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                         action,
                                                                         _env)
                    

                    # print(self.train_dataset[sample_episode,timestep,:])   
                    # break
                    if _info == "False":  
                        print(episode," ",timestep," Solver solution is not feasible")
                        _new_env, (_reward_without_Q, _r), _done, _info = self.jl.actor_solve_and_step(env,
                                                                                  self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                                  self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                                  action,
                                                                                  _env)
                        _new_env, a, (_reward_without_Q, _r), done, info = self.jl.opf_solve_and_step(env,
                                                                                self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                                self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                                _env)
                        break
                        
                    else:
                        new_env = self.update_env(timestep,
                                                  new_env,
                                                  sample_episode,
                                                  sample_episode_list,
                                                  e_)
                        solver_reward += _r
                        not_legal_times += 1

                        #存储正确动作
                        if timestep < 23:
                            state_ = self.normal_state(new_env,
                                                       new_flows,
                                                       timestep+1,
                                                       normal=self.normal)
                        else:
                            state_ = self.normal_state(new_env,
                                                       new_flows,
                                                       25,
                                                       normal=self.normal)
                            
                        env = new_env
                        r = np.array(_r)
                        a = np.array(a).reshape(-1)

                        if timestep+1 == 24:
                            done_bool = 0
                        else:
                            done_bool = 1. - float(done)

                        self.memorize(state, 
                                      a, 
                                      r, 
                                      done_bool, 
                                      state_)

                # if _info == "False":   
                #     break  
                # break
                #存储正确动作

                # agent_reward += r_agent  #记录当前EP的总reward
                agent_reward += r_agent_without_Q  #记录当前EP的总reward
                if self.buffer.count > 1:

                    logging.info(f'Timestep {timestep} : Agent cirtic training')
                    if (timestep+1) % (2*self.freq) == 0 :
                        # 每交互1次，抽取batch_size，进行update_times次critic的训练                                  
                        k = 1 + (self.buffer.count) / self.buffer_size

                        batch_size   = int(k * self.basic_batch_size)
                        update_times = int(k * self.basic_update_times)

                        for index_ in range(update_times):

                            # Sample experience from buffer
                            states, actions, rewards, dones, new_states, _ = self.sample_batch(batch_size)

                            # Predict target q-values using target networks
                            # print(new_states)
                            next_action = self.actor_target.predict(new_states)
                            target_Q1, target_Q2 = self.critic_target.predict([new_states,
                                                                            next_action])
                            target_Q = tf.math.minimum(target_Q1, 
                                                    target_Q2)

                            # Compute critic target
                            critic_target_Q = self.bellman(rewards, 
                                                        target_Q, 
                                                        dones)

                            # Train critic
                            critic_loss = self.critic.train(states, 
                                                            actions, 
                                                            critic_target_Q)
                            _critic_loss += critic_loss
                            
                            # print("critic.model: ",self.critic.model([states,actions])[0][0],self.critic.model([states,actions])[1][0])
                            # print("Actual: ",critic_target[0])
                            # print(self.critic_target.model.get_weights()[0][0])

                    logging.info(f'Timestep {timestep} : Agent actor training')
                    # critic每训练4次，才对actor进行训练    
                    if (timestep+1) % (4*self.freq) == 0 :

                        # Q-Value Gradients under Current Policy
                        actor_loss= self.actor.train(states, 
                                                    self.critic)

                        # print(self.critic_target.model.get_weights()[0][0])
                        _actor_loss += actor_loss

                        if (episode+1) % (4) == 0 :
                            actor_loss= self.actor.supervised_train(states,
                                                                actions)

                        self.critic_transfer_weights()
                        self.actor_transfer_weights()

                    # 使用 TensorBoard 记录数据
                    with summary_writer.as_default():
                        tf.summary.scalar('agent_reward', float(agent_reward), step=episode)
                        tf.summary.scalar('solver_reward', float(solver_reward), step=episode)
                        tf.summary.scalar('not_legal_times', int(not_legal_times), step=episode)
                        tf.summary.scalar('_actor_loss', float(_actor_loss), step=episode)
                        tf.summary.scalar('_critic_loss', float(_critic_loss), step=episode)
                        tf.summary.scalar('First Wrong Timestamp', float(first_wrong_timestamp), step=episode)

                        # if _action_loss < 0.01:
                        #     print(actions)
                        #     print(self.actor.model.predict(states))
                        # if float(_actor_loss)/5 > 24:
                        #     print("target_Q: ", target_Q)
                
            # if _info == "False":   
            #     break  
            # break
            if not_legal_times == 0:
                first_wrong_timestamp = 24
            if self.buffer.count > 1:
                logging.info(f'Episode {episode} : agent_reward {str(round(agent_reward, 2))}, solver_reward {str(round(solver_reward, 2))},not_legal_times {not_legal_times},_actor_loss  {str(round(float(_actor_loss)/3, 2))},_critic_loss {str(round(float(_critic_loss)/6/update_times, 2))}')
                reward_list.append([agent_reward,solver_reward,not_legal_times,float(_actor_loss)/3,float(_critic_loss)/6/update_times])
                tqdm_e.set_description("Score: " + str(round(agent_reward, 5))+" "+
                                    "Solver Score: " + str(round(solver_reward, 5))+" "+
                                    "First Wrong Timestamp: " +str(first_wrong_timestamp) + " "+
                                    "True Score: " + str(r_true)+" "+
                                    "Agent Fail Times: " + str(not_legal_times)+" "+
                                    "Agent Actor Loss: " + str(round(float(_actor_loss)/3, 5))+" "+
                                    "Agent Critic Loss: " + str(round(float(_critic_loss)/6/update_times, 5)))
                tqdm_e.refresh()  
                
                if (self.load_weight_episode+episode+1) % 200 == 0: 
                    # Periodic checkpoint for crash recovery (merge-if-exists)
                    _rl_path = "./"+self.case_name+"/"+self.algo_name+"/reward_list_"+\
                               str(self.act_dim)+"_"+str(self.env_dim)+"_"+\
                               str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+\
                               str(int(self.expand_states))+"_"+str(int(self.noise))+"_"+\
                               str(int(self.normal))+".npy"
                    _trl_path = "./"+self.case_name+"/"+self.algo_name+"/true_reward_list_"+\
                                str(self.act_dim)+"_"+str(self.env_dim)+"_"+\
                                str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+\
                                str(int(self.expand_states))+"_"+str(int(self.noise))+"_"+\
                                str(int(self.normal))+".npy"
                    if os.path.exists(_rl_path) and len(reward_list) > 0:
                        _old_rl = np.load(_rl_path, allow_pickle=True)
                        reward_list = np.vstack((_old_rl, np.array(reward_list)))
                        if len(r_true_list) > 0 and os.path.exists(_trl_path):
                            _old_trl = np.load(_trl_path, allow_pickle=True)
                            r_true_list = np.vstack((_old_trl, np.array(r_true_list).reshape(-1,1)))
                    if len(reward_list) > 0:
                        np.save(_rl_path, np.array(reward_list))
                    if len(r_true_list) > 0:
                        np.save(_trl_path, np.array(r_true_list).reshape(-1,1))
                    reward_list = []
                    r_true_list = []
                    self.save_weights(self.load_weight_episode+episode+1)

            logging.info('Training finish!')
            # '''
            # only for test
            # '''

            # tqdm_e.set_description()
            # tqdm_e.refresh()  
    
        # --- Save final results at end of training ---
        out_dir = "./"+self.case_name+"/"+self.algo_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        suffix = (str(self.act_dim)+"_"+str(self.env_dim)+"_"+
                  str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+
                  str(int(self.expand_states))+"_"+str(int(self.noise))+"_"+
                  str(int(self.normal)))
        
        # Merge with previously saved reward_list (periodic checkpoints or resume)
        rl_path = out_dir+"/reward_list_"+suffix+".npy"
        trl_path = out_dir+"/true_reward_list_"+suffix+".npy"
        if os.path.exists(rl_path) and len(reward_list) > 0:
            old_rl = np.load(rl_path, allow_pickle=True)
            reward_list = np.vstack((old_rl, np.array(reward_list)))
        if len(reward_list) > 0:
            np.save(rl_path, np.array(reward_list))
        if len(r_true_list) > 0:
            r_true_list_arr = np.array(r_true_list).reshape(-1,1)
            if os.path.exists(trl_path):
                old_trl = np.load(trl_path, allow_pickle=True)
                r_true_list_arr = np.vstack((old_trl, r_true_list_arr))
            np.save(trl_path, r_true_list_arr)
        self.save_weights(self.load_weight_episode + self.nb_episodes)
        # Clean up intermediate checkpoints (keep only final)
        import glob as _glob
        _final_ep = str(self.load_weight_episode + self.nb_episodes)
        for _f in _glob.glob(out_dir+"/actor_*.h5") + _glob.glob(out_dir+"/critic_*.h5"):
            if _final_ep + "_" not in _f:
                os.remove(_f)
        logging.info(f'Final weights and rewards saved to {out_dir}')
        print(f'[OK] Saved: {rl_path} ({len(reward_list)} episodes)')
        print(f'[OK] Saved: {out_dir}/actor_*.h5, critic_*.h5')

        return reward_list

    def save_weights(self,episode=100):        
        self.actor.model.save_weights("./"+self.case_name+
                                      "/"+self.algo_name+
                                      "/actor_"+str(episode)+"_"+
                                      str(self.act_dim)+"_"+
                                      str(self.env_dim)+"_"+
                                      str(self.gamma)[2:]+"_"+
                                      str(self.lr)[2:]+"_"+
                                      str(int(self.expand_states))+"_"+
                                      str(int(self.noise))+"_"
                                      +str(int(self.normal))+".h5")
        self.critic.model.save_weights("./"+self.case_name+
                                       "/"+self.algo_name+
                                       "/critic_"+str(episode)+"_"+
                                      str(self.act_dim)+"_"+
                                      str(self.env_dim)+"_"+
                                      str(self.gamma)[2:]+"_"+
                                      str(self.lr)[2:]+"_"+
                                      str(int(self.expand_states))+"_"+
                                      str(int(self.noise))+"_"
                                      +str(int(self.normal))+".h5")

    def load_weights(self,episode=100):
        self.actor.model.load_weights("./"+self.case_name+
                                      "/"+self.algo_name+
                                      "/actor_"+str(episode)+"_"+
                                      str(self.act_dim)+"_"+
                                      str(self.env_dim)+"_"+
                                      str(self.gamma)[2:]+"_"+
                                      str(self.lr)[2:]+"_"+
                                      str(int(self.expand_states))+"_"+
                                      str(int(self.noise))+"_"+
                                      str(int(self.normal))+".h5")
        
        _W =self.actor.model.get_weights()
        self.actor_target.model.set_weights(_W)

        self.critic.model.load_weights("./"+self.case_name+
                                       "/"+self.algo_name+
                                       "/critic_"+str(episode)+"_"+
                                      str(self.act_dim)+"_"+
                                      str(self.env_dim)+"_"+
                                      str(self.gamma)[2:]+"_"+
                                      str(self.lr)[2:]+"_"+
                                      str(int(self.expand_states))+"_"+
                                      str(int(self.noise))+"_"+
                                      str(int(self.normal))+".h5")
        
        _W =self.critic.model.get_weights()
        self.critic_target.model.set_weights(_W)


