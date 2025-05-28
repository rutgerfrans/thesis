import matplotlib.pyplot as plt
fault_p = [0, 0.01, 0.05, 0.1]
tf_delta_times = [0,
7.83747955,
17.0036641,
47.45824925
]
sam_delta_times = [0,
1.0640164,
2.47855095,
3.523182
]
pt_delta_times = [0,
4.038977,
14.885691,
33.100814
]

plt.figure()
plt.plot(fault_p, tf_delta_times,marker='o',label='TensorFlow')
plt.plot(fault_p, pt_delta_times, marker='o',label='PyTorch')
plt.plot(fault_p, sam_delta_times,marker='o',label='SAM')

plt.xlabel('Fault Probability')
plt.ylabel('Wall-clock Time (s)')
plt.title('Delta Fault Tolerance: Fault Probability per Worker vs Wall-clock Time (s)')
plt.legend()
plt.xticks(fault_p)
plt.show()