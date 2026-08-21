'''
Descripttion: 
Author: JIANG Bozhen
version: 
Date: 2024-04-15 19:57:34
LastEditors: JIANG Bozhen
LastEditTime: 2024-10-28 10:31:58
'''

'''
case 24
bus 24
gen 33
branch 37
'''
import numpy as np
import pandas as pd
import sys
import argparse
import os
from utils.SettingParametersClass import SettingParameters
from TD3.td3 import TD3
from DDPG.ddpg import DDPG
from A2C.a2c import A2C
from PPO.ppo import PPO
import utils
from tqdm import tqdm
import tensorflow as tf

from matplotlib import pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU (no GPU available)

import tensorflow as tf
# GPU config disabled - running on CPU
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True #不允许动态增加占用的GPU内存
# config.gpu_options.per_process_gpu_memory_fraction=0.2 #为本程序分配20%的GPU内容
# sess = tf.compat.v1.Session(config = config)

setting = SettingParameters()


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='TD3',help="Algorithm to train from {A2C, DDPG, TD3, PPO}")
    parser.add_argument('--case', type=str, default='case24',help="IEEE standard case")
    #
    parser.add_argument('--nb_episodes', type=int, default=2000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (experience replay)")
    parser.add_argument('--buffer_size', type=int, default=5000, help="Number of buffer size")
    parser.add_argument('--update_times', type=int, default=2, help="update_times")
    parser.add_argument('--frequency', type=int, default=2, help="actor and critic training frequency")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--gamma', type=float, default=0.95, help="discount rate")
    parser.add_argument('--tau', type=float, default=0.005, help="transfer rate")
    parser.add_argument('--cost_discount', type=float, default=100000000, help="cost discount")
    #
    parser.add_argument('--noise', type=bool, default=False, help="wheather add action noise")
    parser.add_argument('--expand_states', type=bool, default=False, help="wheather expand states")
    parser.add_argument('--soft_constrain', type=bool, default=True, help="wheather the constrains are hard")
    parser.add_argument('--load_weight', type=bool, default=False, help="wheather load weights")
    parser.add_argument('--load_weight_episode', type=int, default=0, help="the load weights episode")
    parser.add_argument('--new_lr', type=float, default=0.0001, help="new learning rate")
    #
    parser.add_argument('--normal', type=bool, default=False, help="wheather the input data normalizes")
    parser.set_defaults()
    return parser.parse_args(args)

def run_task(args=None):

    # # 开始计时，计算预训练的时间（包括数据读取和神经网络预训练过程）
    # time_before = time.time()
    # time_trainning = 0 # 计算训练次数
    # ep_reward_list = []
 
    # Parse arguments
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    # 创建 TensorBoard 日志目录
    log_dir = "./tensorboard/"+args.type
    summary_writer = tf.summary.create_file_writer(log_dir)    

    if args.case == "case24":
        if args.expand_states :
            s_dim = 201
        else:
            s_dim = 49

        a_dim = 44 # 33 (P) + 11(V) 
        gen_num = 33  # 33
        bus_num = 24 

    elif args.case == "case118":
        if args.expand_states :
            s_dim = 981
        else:
            s_dim = 237

        a_dim = 108 # 54+49
        gen_num = 54
        bus_num = 118

    elif args.case == "case1354":
        if args.expand_states :
            s_dim = 1354*2+1991*2+1
        else:
            s_dim = 1354*2+1

        a_dim = 260*2 #
        gen_num = 260
        bus_num = 1354

    if args.type == "TD3":
        algo = TD3(a_dim, s_dim, gen_num, bus_num, args)
    elif args.type == "DDPG":
        algo = DDPG(a_dim, s_dim, gen_num, bus_num, args)
    elif args.type == "A2C":
        algo = A2C(a_dim, s_dim, gen_num, bus_num, args)
    elif args.type == "PPO":
        algo = PPO(a_dim, s_dim, gen_num, bus_num, args)

    # Train
    stats = algo.train(summary_writer)
    # Close summary_writer
    summary_writer.close()
    print(f"Training complete. {len(stats)} episodes recorded.")

if __name__ == '__main__':

    run_task()



    


