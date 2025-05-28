import matplotlib.pyplot as plt
neurons = [16,32,64,128,256]
tf_times = [62.0831,
84.457627,
112.126162,
347.505656,
1610.11463
]
pt_times = [52.458581,
65.872381,
91.02153,
271.241718,
1539.059695
]
sam_times = [102.868706,
115.496484,
138.061732,
314.449382,
1426.025175
]

plt.figure()
plt.plot(neurons, tf_times, marker='o', label='TensorFlow')
plt.plot(neurons, pt_times, marker='o', label='PyTorch')
plt.plot(neurons, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Neurons in a Hidden-Layer')
plt.ylabel('Wall-clock Time (s)')
plt.title('Horizontal Scaling: Scaling Model Width vs Wall-clock Time (s)')
plt.legend()
plt.xticks(neurons)
plt.show()