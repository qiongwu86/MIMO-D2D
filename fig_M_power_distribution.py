import os
import numpy as np
import matplotlib.pyplot as plt
from helper import *


def read_log(dir_path):
    fileList = os.listdir(dir_path) #列出文件夹下所有的目录与文件
    fileList = [name for name in fileList if '.npz' in name]

    avg_step_reward = []
    avg_step_sum_power = []
    avg_step_buffer = []
    avg_step_offload_power = []
    avg_step_local_power = []
    avg_step_vehicle_power = []

    for name in fileList[:]:
        path = dir_path + name
        res = np.load(path)

        temp_r = np.array(res['arr_0'])
        temp_sp = np.array(res['arr_1'])
        temp_b = np.array(res['arr_2'])
        temp_op = np.array(res['arr_3'])
        temp_lp = np.array(res['arr_4'])
        temp_vp = np.array(res['arr_5'])

        avg_step_reward.append(temp_r)
        avg_step_sum_power.append(temp_sp)
        avg_step_buffer.append(temp_b)
        avg_step_offload_power.append(temp_op)
        avg_step_local_power.append(temp_lp)
        avg_step_vehicle_power.append(temp_vp)

    avg_step_reward = moving_average(np.mean(avg_step_reward, axis=0, keepdims=True)[0])
    avg_step_sum_power = moving_average(np.mean(avg_step_sum_power, axis=0, keepdims=True)[0])
    avg_step_buffer = moving_average(np.mean(avg_step_buffer, axis=0, keepdims=True)[0])
    avg_step_offload_power = moving_average(np.mean(avg_step_offload_power, axis=0, keepdims=True)[0])
    avg_step_local_power = moving_average(np.mean(avg_step_local_power, axis=0, keepdims=True)[0])
    avg_step_vehicle_power = moving_average(np.mean(avg_step_vehicle_power, axis=0, keepdims=True)[0])

    return avg_step_reward, avg_step_sum_power, avg_step_buffer, avg_step_offload_power, avg_step_local_power,avg_step_vehicle_power

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

[step_r, step_p, step_b, step_lp, step_op,step_vp]= read_log('test_C_ddpg_sigma0_02_rate3_lane2/step_result/')

x = []
for i in range(step_lp.shape[0]):
    dis = np.array([i*0.5 - 250,15,10])
    x.append(i*0.5 - 250)


fig = plt.figure(figsize=(6, 4.5))
plt.plot(x, step_lp+step_vp, color='#00FF00', label='Local-and-V2V', lw=1 )
plt.plot(x, step_op, color='darkred', label='V2I', lw=1 )
#plt.plot(x, step_vp, color='#FF4500', label='V2V', lw=1 )
#print(step_lp,step_op,step_vp)
plt.grid(linestyle=':')
plt.legend()
plt.ylabel('Power')
plt.xlabel('$d_{k,j}(n)$')
plt.show()
