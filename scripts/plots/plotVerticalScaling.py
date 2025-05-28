import matplotlib.pyplot as plt
hiddenlayers = [2,3,4,5]
tf_times = [
38657.8634121041,
31765.89551514,
26906.113437811,
23447.0033635508
]
sam_times = [
48340.822132362,
39711.1569289615,
33803.3572790213,
29365.4313812459
]
pt_times = [
45750.3797138546,
37754.8469869202,
32130.2126132478,
27989.7597131609
]

plt.figure()
plt.plot(hiddenlayers, tf_times, marker='o', label='TensorFlow')
plt.plot(hiddenlayers, pt_times, marker='o', label='PyTorch')
plt.plot(hiddenlayers, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Hidden-Layers')
plt.ylabel('Throughput img/s')
plt.title('Vertical Scaling: Scaling Model Depth vs Wall-clock Time (s)')
plt.legend()
xaxis = [0,2,3,4,5]
plt.xticks(xaxis)
plt.show()