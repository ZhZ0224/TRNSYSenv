import os
from typing import Any, Optional, Tuple
import pandas as pd
import gym
import numpy as np
from gym.spaces import Box
import sys
from gym import spaces

sys.path.append(r'C:\Users\Administrator\Desktop\model\src')

from dataset import updateFile, TOPSIS, tank, Old_step_end
#from src.dataset import updateFile,old_step_start,TOPSIS,tank,old_step_end
import time

MAX_Qsolar=550000
MAX_CONSU = 550000      #实际最大能耗值
MAX_P_HEAT = 5000          #实际最大辅助热源功率
MAX_SOLAR = 5000        #实际最大太阳辐射
MAX_TANK = 80                #最 高 水 箱 温 度
MAX_OUT_TEMPERATURE = 26.75  #实 际 最 高 温 度
MAX_room = 30            # 室 内 最 高 温 度
INITIAL_T_TANK=20        # 初 始 水 箱 温 度
INITIAL_T_room=15    #初始房间温度
INITIAL_TEMPERATURE=-5.2 #初始外界温度
INITIAL_S_solar=1        #初始源端启停
INITIAL_S_HEAT=0         #初始辅助热源启停
MAX_STEPS=672           #最大模拟步数
a1 = [0,0,0]
a = np.zeros((2, 3))
current_step = 0
#dck文件名字
dck_file = r"D:\TRNSYS18\MyProjects\gym\gym.dck"
#控制信号文件
txt_file = r"D:\TRNSYS18\MyProjects\gym\juecebianliang.txt"
#读取状态的文件
xls_flie = r'D:\TRNSYS18\MyProjects\gym\1111.xls'
class TRNSYSEnv(gym.Env):
    #这里没加metadata，这个是可视化环境函数对于trnsys没什么必要
    def __init__(self):
          super().__init__()
          self.reward_range = (0,96) #设置奖励区间，
          #动作空间
          self.continuous_action_space = spaces.Box(low=0, high=1.0, shape=(1,))
          #末端水泵启停
          self.discrete_action_space_1 =  spaces.Discrete(2)
          # 集 热启停
          self.discrete_action_space_2 =  spaces.Discrete(2)
          #辅助热源启停
          self.action_space = spaces.Tuple((self.discrete_action_space_1,self.discrete_action_space_2,self.continuous_action_space))

    #状态空间，包含最近24h的室外温度，室内温度，水箱温度,太阳辐射值
          self.observation_space = spaces.Box(low=-1,high=5,shape=(9,),dtype=np.float32)
#每回合进行重置
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        #更改结束时间重置为0.25，开始时间不用因为是0，
        self.step_end = 'STOP='+str(0.25)
        print(self.step_end)
        #读取旧的结束时间，然后让其进行更改                  
        self.old_step_end = Old_step_end(dck_file)
        updateFile(dck_file, self.old_step_end, self.step_end)
        self.current_step= 0                             #重 置 步 长
        #初始动作值为0，0，0
        pinjie=np.zeros((1, 3))
        #将动作值放在txt文件里面方便软件读取
        np.savetxt(txt_file, np.c_[pinjie],fmt='%.18e',delimiter='\t')
        #初始化水箱温度
        tank(dck_file)
        super().reset(seed=seed, options=options)
        obs = self._next_observation()
        return obs,{}
#观测值
    def _next_observation(self):
        #运存不够时加入休息时间
        time.sleep(0.2)
        #运行程序
        os.system(r'D:\TRNSYS18\Exe\TrnEXE64.exe  D:\TRNSYS18\MyProjects\gym\gym.dck /h')
        #os.system(r'D:\TRNSYS18\Exe\TrnEXE64.exe  D:\TRNSYS18\MyProjects\gym\gym.dck /h')
        #运行完了之后读取文件内容
        df = pd.read_csv(xls_flie, delimiter='\t', header= None, usecols=range(1, 13))
        self.df = df #接受dataframe数据
        obs = np.array([
                float(self.df.iloc[-1, 0])/4000,              #制造太阳辐射{0，5}的索引，将数归一化（除以最高太阳辐射值）
                float(self.df.iloc[-1, 1])/10,                #室外温度
                float(self.df.iloc[-1, 2])/13500,             #辅热能耗   
                float(self.df.iloc[-1, 3])/100,               #水箱温度
                float(self.df.iloc[-1, 4])/27000,             #得热两
                float(self.df.iloc[-1, 5])/800,               #热损失
                float(self.df.iloc[-1, 6])/25,                #房间温度
                                    
                float(self.df.iloc[-1, 8])/5 ,              #温差
                float(self.df.iloc[-1, 9])/5,               #能耗
            ])
        self.SOLAR = float(self.df.iloc[-1, 0])
        self.CONSU = float(self.df.iloc[-1, 9])
        self.RE = float(self.df.iloc[-1, 4])/(float(self.df.iloc[-1, 0])+float(self.df.iloc[-1, 4])+0.0001)
        self.room_1 = float(self.df.iloc[-1, 8])
        self.Qsolar = float(self.df.iloc[-1, 4])
        self.room = float(self.df.iloc[-1, 7])
        self.Ttank = float(self.df.iloc[-1, 3])
        self.loss = float(self.df.iloc[-1, 5])
        print('room',self.room)
        return obs
#动作拼接
    def take_action(self, action):
        a1=pd.read_csv(txt_file, delimiter='\t',header=None)
        self.action = np.array(action)
        a=np.array(a1) 
        self.pinjie=np.vstack((a,self.action))  
    #将随机值带入控制信号文件
        np.savetxt(txt_file, np.c_[self.pinjie],fmt='%.18e',delimiter='\t')  #np.c_[i]是保存的数据，fmt='%.18e'科学计数法。delimiter='\t'数据之间的空格
    #进行trnsys运算
    def step(self, action):
        # action 是 1 * n 的矩阵 n表示模型数量，和为1
        self.action = action
        self.action = np.array(self.action, dtype=object).reshape(-1, 3)
        # 拼接动作值
        self.take_action(self.action)

        #########步长改变，从第一步转化到下一步#####################
        obs = self._next_observation()
        self.old_step_end = Old_step_end(dck_file)
        ########################分开模拟时间并进行修改
        self.number = float(self.old_step_end.split("=")[1])+0.25

        self.step_end = "STOP=" + str(self.number)

        updateFile(r"D:\TRNSYS18\MyProjects\gym\gym.dck", self.old_step_end, self.step_end)
        self.current_step += 1     #步骤加一       
        #计算奖励值
        reward =TOPSIS(self.Qsolar,self.CONSU,self.room_1)
        
        if self.current_step >= 96:
            done = True
        else:                   
            done = False
        print(self.current_step,done,reward)
        return obs, reward, done,{},{}   

