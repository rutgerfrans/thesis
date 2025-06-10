import matplotlib.pyplot as plt
datasetsizes = [10000,30000,60000]
tf_times = [ 
8412.4681281234,
26313.4338927739,
48211.9536545212
]
pt_times = [
12522.5258180254,
24036.6852721998,
47026.277574929
]
sam_times = [
8848.14447482155,
28142.6941952573,
54780.6753250888
]

plt.figure()
plt.plot(datasetsizes, tf_times, marker='o', label='TensorFlow')
plt.plot(datasetsizes, pt_times, marker='o', label='PyTorch')
plt.plot(datasetsizes, sam_times, marker='o', label='SAM')

plt.xlabel('Images')
plt.ylabel('Images/s')
plt.title('Throughput vs. Dataset Size')
plt.legend()
xaxis = [10000,30000,60000]
plt.xticks(xaxis)
plt.ylim(bottom=0)
plt.savefig('plots/images/weakscalingresult.png', dpi=300,bbox_inches='tight')
#plt.show()