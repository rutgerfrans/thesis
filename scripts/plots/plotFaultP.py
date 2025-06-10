import matplotlib.pyplot as plt
fault_p = [0, 0.01, 0.05, 0.1]
tf_delta_times = [
50474.8087687404,
44523.5850591301,
33488.7196230678,
21551.1881187193
]
pt_delta_times = [
47026.277574929,
44849.5011906444,
43841.0667537293,
31966.205148033
]
fault_p_sam = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
sam_delta_times = [
53088.8668489293,
46862.9221801578,
45227.2492322307,
45341.737101875
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
plt.savefig('plots/images/faultpresult.png', dpi=300,bbox_inches='tight')
#plt.show()