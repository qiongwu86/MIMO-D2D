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
		temp_rs = np.array(res['arr_1'])   #0:reward  1:power 2:buffer
		avg_rs.append(temp_rs)
	avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
	return avg_rs

#ddpg_avg_power = np.mean(output_avg('test_S_ddpg_sigma0_02_rate3_lane2/'), axis=0)
#GD_local_avg_power = np.mean(output_avg('test_S_GD_Local_lane2/'), axis=0)
#GD_offload_avg_power = np.mean(output_avg('test_S_GD_Offload_lane2/'), axis=0)

ddpg_avg_power_m = np.mean(output_avg('test_C_ddpg_sigma0_02_rate3_lane2/'), axis=0)
#ddpg_avg_power_m1 = np.mean(output_avg('test_M_ddpg_sigma0_02_rate3_lane2/'), axis=0)
GD_local_avg_power_m = np.mean(output_avg('test_C_GD_Local_lane2_rate_3/'), axis=0)
GD_offload_avg_power_m = np.mean(output_avg('test_C_GD_Offload_lane2_rate_3/'), axis=0)
GD_vehicle_avg_power_m = np.mean(output_avg('test_C_GD_vehicle_lane2_rate_3/'), axis=0)

#power = [ddpg_avg_power_m,ddpg_avg_power_m1]
# power_M = [ddpg_avg_power_m,GD_local_avg_power_m,GD_offload_avg_power_m]
#labels = ['DDPG', 'DDPG1']

#name = ["Single user","Multiple users"]
#y1 = [ddpg_avg_power,ddpg_avg_power_m]
#y2 = [GD_local_avg_power,GD_local_avg_power_m]
#y3 = [GD_offload_avg_power,GD_offload_avg_power_m]


name = [""]
y1 = [ddpg_avg_power_m]
#y2 = [ddpg_avg_power_m1]
y3 = [GD_local_avg_power_m]
y4 = [GD_offload_avg_power_m]
y5 = [GD_vehicle_avg_power_m]


x = np.arange(len(name))
width = 0.25



font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=22) 
font1 = {'family' : 'SimSun',
'weight' : 'normal',
'size'   : 22,
}

figure, ax = plt.subplots(figsize=(5, 4))

#设置坐标刻度值的大小以及刻度值的字体
# plt.tick_params(labelsize=22)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]


# plt.bar(x, y1,  width=width, label='最优策略',color='0.25')
# plt.bar(x + width, y2, width=width, label='本地贪婪策略', color='0.5', tick_label=name)
# plt.bar(x + 2 * width, y3, width=width, label='卸载贪婪策略', color='0.75')
# plt.legend(prop=font1)


# plt.bar(x, y1,  width=width, label='DDPG',color='0.25')
# plt.bar(x + width, y2, width=width, label='GD-Local', color='0.5', tick_label=name)
# plt.bar(x + 2 * width, y3, width=width, label='GD-offload', color='0.75')
# plt.legend(prop=font1)


plt.bar(x, y1,  width=width, label='DDPG',color='#1f77b4')
#plt.bar(x + width, y2, width=width, label='DDPG1', color='salmon', tick_label=name)
plt.bar(x + 1 * width, y3, width=width, label='GD-Local', color='#00FF00')
plt.bar(x + 2 * width, y4, width=width, label='GD-V2I', color='darkred')
plt.bar(x + 3 * width, y5, width=width, label='GD-V2V', color='#FF4500')

# 显示在图形上的值
# for a, b in zip(x,y1):
#     # plt.text(a, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x,y2):
#     plt.text(a+width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, y3):
#     plt.text(a+2*width, b+0.1, b, ha='center', va='bottom')

plt.xticks()
# plt.ylabel('平均计算任务缓存长度',fontproperties=font)
# plt.xlabel("(b)",fontproperties=font)

plt.xlabel('policies')

# fig = plt.figure(figsize=(6, 4.5))

# plt.bar(range(len(power)), power,tick_label=labels)
# plt.bar(range(len(power_M)), power_M,tick_label=labels)

plt.grid(linestyle=':')

plt.ylabel('average power')
#plt.ylim([0,100])
#plt.ylabel('average buffer capacity')
#plt.ylabel('average reward')
plt.legend()
plt.xticks([])
plt.show()