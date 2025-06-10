import matplotlib.pyplot as plt
neurons = [16,32,64,128,256]
tf_times = [
50474.8087687404,
41153.2711074497,
24546.842186979,
8657.03349468421,
1600.63823321021
]
pt_times = [
47026.277574929,
42052.2076087443,
25342.8848912902,
8483.05737215349,
1585.0629729751
]
sam_times = [
53088.8668489293,
39695.8683428563,
26684.2914282737,
8381.25584733021,
1489.28444023928
]

plt.figure()
plt.plot(neurons, tf_times, marker='o', label='TensorFlow')
plt.plot(neurons, pt_times, marker='o', label='PyTorch')
plt.plot(neurons, sam_times, marker='o', label='SAM')

plt.xlabel('# Neurons in Hidden-Layer')
plt.ylabel('Images/s')
plt.title('Throughput vs. Model Width')
plt.legend()
plt.xticks(neurons)
plt.ylim(bottom=0)
plt.savefig('plots/images/horizontalscalingresult.png', dpi=300,bbox_inches='tight')
#plt.show()