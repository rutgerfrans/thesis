import matplotlib.pyplot as plt
fault_p = [0, 0.01, 0.05, 0.1]
tf_delta_times = [
38657.8634121041,
35510.5429437063,
24781.7921255004,
17247.2947297525
]
fault_p_sam = [0, 0.01, 0.05, 0.1]
sam_delta_times = [
48340.822132362,
46862.9221801578,
45227.2492322307,
45341.737101875
]
pt_delta_times = [
45750.3797138546,
44225.3954058084,
37092.243661406,
26562.3897634885
]

plt.figure()
plt.plot(fault_p, tf_delta_times,marker='o',label='TensorFlow')
plt.plot(fault_p, pt_delta_times, marker='o',label='PyTorch')
plt.plot(fault_p, sam_delta_times,marker='o',label='SAM')

plt.xlabel('p')
plt.ylabel('Images/s')
plt.title('Throughput vs. Fault Probality')
plt.legend()
plt.xticks(fault_p)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.savefig('plots/images/faultpresultext.png', dpi=300,bbox_inches='tight')
#plt.show()