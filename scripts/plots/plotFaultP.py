import matplotlib.pyplot as plt
fault_p = [0, 0.01, 0.05, 0.1]
tf_delta_times = [
38657.8634121041,
34324.6582829561,
30346.4179791875,
21909.5347686709
]
sam_delta_times = [
48340.822132362,
46633.8385234381,
45227.2492322307,
45341.737101875
]
pt_delta_times = [
45750.3797138546,
42479.7121319828,
35637.7744716274,
28050.6890074132
]

plt.figure()
plt.plot(fault_p, tf_delta_times,marker='o',label='TensorFlow')
plt.plot(fault_p, pt_delta_times, marker='o',label='PyTorch')
plt.plot(fault_p, sam_delta_times,marker='o',label='SAM')

plt.xlabel('Fault Probability')
plt.ylabel('Throughput img/s')
plt.title('Delta Fault Tolerance: Fault Probability per Worker vs Wall-clock Time (s)')
plt.legend()
xaxis = [0, 0.01, 0.05, 0.1]
plt.xticks(xaxis)
plt.show()