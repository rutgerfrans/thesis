import matplotlib.pyplot as plt
fault_p = [0, 0.01, 0.05, 0.1]
tf_delta_times = [0,12.924263,20.774263,39.293493]
sam_delta_times = [0,5.747806,6.647806,5.247806]
pt_delta_times = [0,40.95,54.6,93.15]

plt.figure()
plt.plot(fault_p, tf_delta_times,marker='o',label='TensorFlow')
plt.plot(fault_p, pt_delta_times, marker='o',label='PyTorch')
plt.plot(fault_p, sam_delta_times,marker='o',label='SAM')

plt.xlabel('Fault Probability')
plt.ylabel('Wall-clock Time (s)')
plt.title('Fault Tolerance: Fault Probability per Worker vs Wall-clock Time (s)')
plt.legend()
plt.xticks(fault_p)
plt.show()