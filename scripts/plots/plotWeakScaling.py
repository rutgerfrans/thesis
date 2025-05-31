import matplotlib.pyplot as plt
datasetsizes = [10000,30000,60000]
tf_times = [ 
6299.81380900288,
19050.5371060805,
37730.0311569881
]
pt_times = [
10648.471410598,
22469.4199620349,
45613.3880311344
]
sam_times = [
8056.803688727,
24104.195204611,
48425.8999792091
]

plt.figure()
plt.plot(datasetsizes, tf_times, marker='o', label='TensorFlow')
plt.plot(datasetsizes, pt_times, marker='o', label='PyTorch')
plt.plot(datasetsizes, sam_times, marker='o', label='SAM')

plt.xlabel('Dataset Size)')
plt.ylabel('Throughput images/s')
plt.title('Throughput (images/s) vs. Dataset Size')
plt.legend()
xaxis = [10000,30000,60000]
plt.xticks(xaxis)
plt.ylim(bottom=0)
plt.savefig('plots/images/weakscalingresult.png', dpi=300,bbox_inches='tight')
#plt.show()