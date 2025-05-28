import matplotlib.pyplot as plt
neurons = [16,32,64,128,256]
tf_times = [
38657.8634121041,
28416.6165360057,
21404.4604505414,
6906.3624104006,
1490.57710257561
]
pt_times = [
45750.3797138546,
36434.0860853352,
26367.3880234709,
8848.19642677532,
1559.39370499856

]
sam_times = [
48340.822132362,
38422.1256115662,
26850.6923618842,
8047.72720715659,
1498.94626232494
]

plt.figure()
plt.plot(neurons, tf_times, marker='o', label='TensorFlow')
plt.plot(neurons, pt_times, marker='o', label='PyTorch')
plt.plot(neurons, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Neurons in a Hidden-Layer')
plt.ylabel('Throughput img/s')
plt.title('Horizontal Scaling: Scaling Model Width vs Wall-clock Time (s)')
plt.legend()
plt.xticks(neurons)
plt.show()