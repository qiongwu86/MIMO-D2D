import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def output_avg(dir):
	dir_path = dir
	fileList = os.listdir(dir_path) 
	fileList = [name for name in fileList if '.npz' in name]
	avg_rs = []
	for name in fileList:
		path = dir_path + name
		res = np.load(path)
		temp_rs = np.array(res['arr_0'])
		avg_rs.append(temp_rs)
	avg_rs = moving_average(np.mean(avg_rs, axis=0, keepdims=True)[0],30)
	return avg_rs

ddpg_reward = output_avg('test_C_ddpg_sigma0_02_rate3_lane2/step_result/')
#ddpg_reward1 = output_avg('test_M_ddpg_sigma0_02_rate3_lane2/step_result/')
GD_local_reward = output_avg('test_C_GD_Local_lane2_rate_3/step_result/')
GD_offload_reward = output_avg('test_C_GD_Offload_lane2_rate_3/step_result/')
GD_vehicle_reward = output_avg('test_C_GD_vehicle_lane2_rate_3/step_result/')

x = []
for i in range(ddpg_reward.shape[0]):
    x.append(i*0.5 - 250)

fig = plt.figure(figsize=(6, 4.5))
plt.plot(x, ddpg_reward, color='#1f77b4', label='DDPG',lw=1 )
#plt.plot(x, ddpg_reward1, color='salmon', label='DDPG1',lw=1 )
plt.plot(x, GD_local_reward, color='#00FF00', label='Greedy_Local',lw=1)
plt.plot(x, GD_offload_reward, color='darkred',label='Greedy_V2I',lw=1)
plt.plot(x, GD_vehicle_reward, color='#FF4500',label='Greedy_V2V',lw=1)


plt.grid(linestyle=':')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('$d_{k,j}(n)$')
plt.show()
# fig.savefig('figs/buffer.eps', format='eps', dpi=1000)
