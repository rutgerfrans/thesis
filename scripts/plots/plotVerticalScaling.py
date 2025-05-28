import matplotlib.pyplot as plt
hiddenlayers = [0,2,3,4,5]
tf_times = [0,
62.0831,
75.552726,
89.199059,
102.358496
]
sam_times = [0,
102.868706,
115.479359,
128.149642,
141.236291
]
pt_times = [0,
52.458581,
63.567997,
74.696051,
85.745645
]

plt.figure()
plt.plot(hiddenlayers, tf_times, marker='o', label='TensorFlow')
plt.plot(hiddenlayers, pt_times, marker='o', label='PyTorch')
plt.plot(hiddenlayers, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Hidden-Layers')
plt.ylabel('Wall-Clock Time (s)')
plt.title('Vertical Scaling: Scaling Model Depth vs Wall-clock Time (s)')
plt.legend()
plt.xticks(hiddenlayers)
plt.show()