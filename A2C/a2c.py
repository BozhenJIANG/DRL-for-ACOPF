'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-07-28 18:04:44
LastEditors: JIANG Bozhen
LastEditTime: 2024-10-29 15:05:42
'''
import numpy as np
import os
import logging

from tqdm import tqdm
from .actor import Actor
from .critic import Critic
from utils.SettingParametersClass import SettingParameters
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from utils.utilize import *
# python-Julia接口学习
from julia.api import Julia
j = Julia(compiled_modules=False)

from julia import Main as jl

class A2C:
    """ Actor-Critic Main Algorithm
    """
    def __init__(self, act_dim, env_dim, gen_num, bus_num, args):
        """ Initialization
        """
        self.case_name = args.case
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = args.gamma
        self.lr = args.lr
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, 0.1*args.lr)
        self.critic = Critic(self.env_dim, act_dim, args.lr)
        self.algo_name = args.type
        self.load_weight_episode = args.load_weight_episode

        self.gen_num = gen_num
        self.bus_num = bus_num

        self.expand_states = args.expand_states
        self.noise = args.noise 
        self.nb_episodes =args.nb_episodes
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
        logging.info('Start A2C instantiation')

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

        if args.load_weight:
            self.actor = Actor(self.env_dim, act_dim, 0.1*args.new_lr)
            self.critic = Critic(self.env_dim, act_dim, args.new_lr)
            self.load_weights(args.load_weight_episode)

    def policy_action(self, s):
        """ Use the actor_target to predict value
        """       
        if self.noise:
            mu,sigma = self.actor.predict(s)
            _action = tf.clip_by_value(tf.random.normal([1], mu, sigma, tf.float32), clip_value_min=0, clip_value_max=1)
            return tf.clip_by_value(_action+
                                    tf.random.truncated_normal(_action.shape,
                                                               mean=0,
                                                               stddev=0.1),
                                    clip_value_min=0,
                                    clip_value_max=1).numpy()
        else:
            mu,sigma = self.actor.predict(s)
            return tf.clip_by_value(tf.random.normal([1], mu, sigma, tf.float32), clip_value_min=0, clip_value_max=1).numpy()
    
    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

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
        state_temp = julia_data_to_python_data(env,flows,timestep,self.expand_states,normal)
        if normal:
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

        #First, gather experience
        tqdm_e = tqdm(range(self.nb_episodes), desc='Score', leave=True, unit=" episodes")
        sample_episode_list = np.random.randint(800,size=self.nb_episodes)
        #开始训练
    
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

            actions, states, rewards = [], [], []

            for timestep in range(24):
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
                if sample_episode+1 < self.nb_episodes:
                    if timestep+1 <24 :
                        (new_env,new_flows), (_, r), done, info = self.jl.actor_solve_and_step(env,
                                                    self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                    self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                    action,
                                                    _env) 
                    else:
                        break
                else:
                    if timestep+1 <24 :
                        (new_env,new_flows), (_, r), done, info = self.jl.actor_solve_and_step(env,
                                                    self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                    self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                    action,
                                                    _env)
                    else:
                        break
                r_agent = r
                # break
                # if timestep == 0:                
                #     print("Agent action: ",tf.squeeze(action).numpy())
                #     print("Agentr reward: ", r)
                # print("agent action: ",tf.squeeze(a).numpy())
                # print("Agentr reward: ", r)
                # print("Agent done: ", done)
                # break

                new_env = self.update_env(timestep,
                                          new_env,
                                          sample_episode,
                                          sample_episode_list,
                                          e_)

                if info == "True" :
                    solver_reward += r
                    logging.info(f'Timestep {timestep} : Agent solution is feasible')
                    #记录Agent真实的回报
                    if r_true_flag == 0:
                        r_true += r 

                    #更新环境
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
                    (new_env,new_flows), a, (_, r), done, info = self.jl.opf_solve_and_step(env,
                                                                      self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                      self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                      _env)

                    # check the Solver solution whether fesible
                    action = inverse_a2action(a,
                                               env,
                                               self.gen_num,
                                               PV_bus_set)


                    (_new_env,_new_flows), (_, _r), _done, _info = self.jl.actor_solve_and_step(env,
                                                                         self.train_dataset[sample_episode,timestep,self._non_zero_index][0],
                                                                         self.train_dataset[sample_episode,timestep+24,self._non_zero_index][0],
                                                                         action,
                                                                         _env)
                       
                    # break
                    if _info == "False":  
                        print(episode," ",timestep," Solver solution is not feasible") 

                        break
                        
                    else:
                        solver_reward += r
                        new_env = self.update_env(timestep,
                                                  new_env,
                                                  sample_episode,
                                                  sample_episode_list,
                                                  e_)

                        not_legal_times += 1
                            
                        env = new_env
                        r = np.array(r)
                        a = np.array(a).reshape(-1)

                        if timestep+1 == 24:
                            done_bool = 0
                        else:
                            done_bool = 1. - float(done)

                actions.append(np.reshape(a,[-1]))
                rewards.append(r)
                states.append(state)

                env = new_env
               
                agent_reward += r_agent  #记录当前EP的总reward      
            

            actions = np.array(actions)
            rewards = np.array(rewards)
            states = np.array(states)
            if actions.size != 0:
                logging.info(f'Timestep {timestep} : Agent training')
                # break
                discounted_rewards = self.discount(rewards)
                state_values = self.critic.predict(states)

                advantages = discounted_rewards - np.reshape(state_values, len(state_values))

                _actor_loss = self.actor.train(states, actions, advantages)
                # break
                _critic_loss = self.critic.train(states, discounted_rewards)
                logging.info(f'Episode {episode} : agent_reward {str(round(agent_reward, 2))}, solver_reward {str(round(solver_reward, 2))},not_legal_times {not_legal_times},_actor_loss  {str(round(float(_actor_loss), 2))},_critic_loss {str(round(float(_critic_loss), 2))}')
                reward_list.append([agent_reward,not_legal_times,float(_actor_loss),float(_critic_loss)])
                tqdm_e.set_description("Score: " + str(agent_reward)+" "+
                                    "First Wrong Timestamp: " +str(first_wrong_timestamp) + " "+
                                    "True Score: " + str(r_true)+" "+
                                    "Agent Fail Times: " + str(not_legal_times)+" "+
                                    "Agent Actor Loss: " + str(round(float(_actor_loss), 5))+" "+
                                    "Agent Critic Loss: " + str(round(float(_critic_loss), 5)))
                tqdm_e.refresh()  
             
            if (self.load_weight_episode+episode+1) % 100 == 0: 
                if self.load_weight_episode+episode+1 == 100:
                    np.save("./"+self.case_name+"/"+
                            self.algo_name+
                            "/reward_list_"+
                            str(self.act_dim)+"_"+
                            str(self.env_dim)+"_"+
                            str(self.gamma)[2:]+"_"+
                            str(self.lr)[2:]+"_"+
                            str(int(self.expand_states))+"_"+
                            str(int(self.noise))+".npy",
                            np.array(reward_list))
                    np.save("./"+self.case_name+"/"+self.algo_name+"/true_reward_list_"+
                            str(self.act_dim)+"_"+
                            str(self.env_dim)+"_"+
                            str(self.gamma)[2:]+"_"+
                            str(self.lr)[2:]+"_"+
                            str(int(self.expand_states))+"_"+
                            str(int(self.noise))+".npy",
                            np.array(r_true_list).reshape(-1,1))
                else:
                    pre_reward_list = np.load("./"+self.case_name+"/"+
                                              self.algo_name+
                                              "/reward_list_"+
                                              str(self.act_dim)+"_"+
                                              str(self.env_dim)+"_"+
                                              str(self.gamma)[2:]+"_"+
                                              str(self.lr)[2:]+"_"+
                                              str(int(self.expand_states))+"_"+
                                              str(int(self.noise))+".npy",
                                              allow_pickle=True)
                    pre_r_true_list = np.load("./"+self.case_name+"/"+
                                              self.algo_name+
                                              "/true_reward_list_"+
                                              str(self.act_dim)+"_"+
                                              str(self.env_dim)+"_"+
                                              str(self.gamma)[2:]+"_"+
                                              str(self.lr)[2:]+"_"+
                                              str(int(self.expand_states))+"_"+
                                              str(int(self.noise))+".npy",
                                              allow_pickle=True)
                    # History files were aligned to the resumed checkpoint at
                    # load time, so a plain append keeps episodes aligned.
                    if len(reward_list) > 0:
                        reward_list = np.vstack((pre_reward_list, np.array(reward_list)))
                    else:
                        reward_list = pre_reward_list
                    if len(r_true_list) > 0:
                        r_true_list = np.vstack((pre_r_true_list, np.array(r_true_list).reshape(-1,1)))
                    else:
                        r_true_list = pre_r_true_list

                    np.save("./"+self.case_name+"/"+
                            self.algo_name+
                            "/reward_list_"+
                            str(self.act_dim)+"_"+
                            str(self.env_dim)+"_"+
                            str(self.gamma)[2:]+"_"+
                            str(self.lr)[2:]+"_"+
                            str(int(self.expand_states))+"_"+
                            str(int(self.noise))+".npy",reward_list)
                    
                    np.save("./"+self.case_name+"/"+self.algo_name+"/true_reward_list_"+
                            str(self.act_dim)+"_"+
                            str(self.env_dim)+"_"+
                            str(self.gamma)[2:]+"_"+
                            str(self.lr)[2:]+"_"+
                            str(int(self.expand_states))+"_"+
                            str(int(self.noise))+".npy",
                            r_true_list)
                    
                reward_list = []
                r_true_list = []   

                self.save_weights(self.load_weight_episode+episode+1)


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
                  str(int(self.expand_states))+"_"+str(int(self.noise)))
        rl_path = out_dir+"/reward_list_"+suffix+".npy"
        trl_path = out_dir+"/true_reward_list_"+suffix+".npy"
        if os.path.exists(rl_path):
            old_rl = np.load(rl_path, allow_pickle=True)
            if len(reward_list) > 0:
                reward_list = np.vstack((old_rl, np.array(reward_list)))
            else:
                reward_list = old_rl
        np.save(rl_path, np.array(reward_list))
        if len(r_true_list) > 0:
            new_trl = np.array(r_true_list).reshape(-1,1)
            if os.path.exists(trl_path):
                old_trl = np.load(trl_path, allow_pickle=True)
                new_trl = np.vstack((old_trl, new_trl))
            np.save(trl_path, new_trl)
        self.save_weights(self.load_weight_episode + self.nb_episodes)
        # Clean up intermediate checkpoints (keep only final)
        import glob as _glob
        _final_ep = str(self.load_weight_episode + self.nb_episodes)
        for _f in _glob.glob(out_dir+"/actor_*.h5") + _glob.glob(out_dir+"/critic_*.h5"):
            if _final_ep + "_" not in _f:
                os.remove(_f)
        print(f"[OK] Saved: {rl_path} ({len(reward_list)} episodes)")
        print(f"[OK] Saved: {out_dir}/actor_*.h5, critic_*.h5")

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

        # One-time alignment of the reward history with the resumed checkpoint:
        # a crashed segment may have flushed rows beyond `episode` that are
        # about to be re-trained. Truncate so that later merges (which append
        # without truncating) stay episode-aligned.
        _suffix = (str(self.act_dim)+"_"+str(self.env_dim)+"_"+
                   str(self.gamma)[2:]+"_"+str(self.lr)[2:]+"_"+
                   str(int(self.expand_states))+"_"+str(int(self.noise)))
        _rl_path = "./"+self.case_name+"/"+self.algo_name+"/reward_list_"+_suffix+".npy"
        _trl_path = "./"+self.case_name+"/"+self.algo_name+"/true_reward_list_"+_suffix+".npy"
        if os.path.exists(_rl_path):
            _old_rl = np.load(_rl_path, allow_pickle=True)
            if len(_old_rl) > episode:
                np.save(_rl_path, _old_rl[:episode])
        if os.path.exists(_trl_path):
            _old_trl = np.load(_trl_path, allow_pickle=True)
            if len(_old_trl) > episode:
                np.save(_trl_path, _old_trl[:episode])

