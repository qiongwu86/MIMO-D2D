import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties  

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

def long_term_disc_reward(set):
	r=0
	gamma=0.99
	for i in range(0,set.shape[0]):
		r = r + gamma*set[i]
	return r

ddpg_reward = output_avg('test_C_ddpg_sigma0_02_rate3_lane2/step_result/')
#ddpg_reward1 = output_avg('test_M_ddpg_sigma0_02_rate3_lane2/step_result/')
GD_local_reward = output_avg('test_C_GD_local_lane2_rate_3/step_result/')
GD_offload_reward = output_avg('test_C_GD_Offload_lane2_rate_3/step_result/')
GD_vehicle_reward = output_avg('test_C_GD_vehicle_lane2_rate_3/step_result/')


name = ["Polices"]
y1 = [long_term_disc_reward(ddpg_reward)]
#y2 = [long_term_disc_reward(ddpg_reward1)]
y2 = [long_term_disc_reward(GD_local_reward)]
y3 = [long_term_disc_reward(GD_offload_reward)]
y4 = [long_term_disc_reward(GD_vehicle_reward)]

figure, ax = plt.subplots(figsize=(5.2, 4))

x = np.arange(len(name))
width = 0.25

plt.bar(x, y1,  width=width, label='DDPG',color='#1f77b4')
plt.bar(x + width, y2, width=width, label='GD-Local', color='#00FF00', tick_label="")
plt.bar(x + 2 * width, y3, width=width, label='GD-V2I', color='darkred')
plt.bar(x + 3 * width, y4, width=width, label='GD-V2V', color='#FF4500')

# # 显示在图形上的值
# for a, b in zip(x,y1):
#     plt.text(a, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x,y2):
#     plt.text(a+width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, y3):
#     plt.text(a+2*width, b+0.1, b, ha='center', va='bottom')

plt.xticks()
# plt.grid(linestyle=':')

plt.ylabel('Cumulative Discount Reward')
plt.xlabel('policies')
plt.legend()
plt.show()
# fig.savefig('figs/buffer.eps', format='eps', dpi=1000)


# plt.show()
