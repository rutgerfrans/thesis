import matplotlib.pyplot as plt
datasetsizes = [0, 10000,30000,60000]
tf_times = [0, 
53.977998,
53.135708,
62.0831
]
pt_times = [0,
38.595782,
52.554307,
52.458581
]
sam_times = [0,
21.221085,
53.626296,
102.868706
]

plt.figure()
plt.plot(datasetsizes, tf_times, marker='o', label='TensorFlow')
plt.plot(datasetsizes, pt_times, marker='o', label='PyTorch')
plt.plot(datasetsizes, sam_times, marker='o', label='SAM')

plt.xlabel('Dataset Size')
plt.ylabel('Wall-clock Time (s)')
plt.title('Weak Scaling: Scaling Dataset vs Wall-clock Time (s)')
plt.legend()
plt.xticks(datasetsizes)
plt.show()