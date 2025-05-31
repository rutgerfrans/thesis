import matplotlib.pyplot as plt
neurons = [16,32,64,128,256]
tf_times = [
37807.5585816305,
30578.1187034159,
21977.9324159348,
8297.17875144932,
1541.29296081512
]
pt_times = [
46615.7083166561,
36364.396710113,
26209.7308780465,
8535.71759825862,
1564.84298867179
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

plt.xlabel('# Neurons in Hidden-Layer')
plt.ylabel('Throughput images/s')
plt.title('Throughput (images/s) vs. # Neurons in Hidden-Layer')
plt.legend()
plt.xticks(neurons)
plt.ylim(bottom=0)
plt.savefig('plots/images/horizontalscalingresult.png', dpi=300,bbox_inches='tight')
#plt.show()