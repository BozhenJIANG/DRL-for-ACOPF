'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-03-26 21:24:34
LastEditors: JIANG Bozhen
LastEditTime: 2024-04-15 12:00:26
'''
class SettingParameters:
    def __init__(self):
        ########     主程序参数设置      ########
        # 令电压v的调整值为0
        file_ = "r_0621_288_1"
        self.reward_saving_filename = 'reward_thomas//' + file_
        self.log_filename = 'log/' + file_ + '.log'

        self.train_ahead_time = 2000  # 提前预训练的最大次数
        self.max_action = 1

        self.max_timestep = 288  # 每个回合中最大时间步数
        self.max_episode = 100000  # 回合数
        self.saving_timestep = 2  # 进行数据保存的时间步数，也就是每saving_timestep步对数据进行保存

        ######## actor-critic网络参数设置 ########
        ##    全局参数    ##
        self.BATCH_SIZE = 64
        self.LR_A = 0.00001  # learning rate for actor
        self.LR_C = 0.00001  # learning rate for critic
        self.discount = 0.95  # reward discount
        self.TAU = 0.005  # soft replacement

        self.policy_noise = 0
        self.noise_clip = 0.05
        self.policy_freq = 1





