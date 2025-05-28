import matplotlib.pyplot as plt
datasetsizes = [10000,30000,60000]
tf_times = [ 
7410.42674461546,
22583.6832737789,
38657.8634121041
]
pt_times = [
10363.8268036647,
22833.5234255872,
45750.3797138546
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

plt.xlabel('Dataset Size')
plt.ylabel('Throughput img/s')
plt.title('Weak Scaling: Scaling Dataset vs Wall-clock Time (s)')
plt.legend()
xaxis = [0, 10000,30000,60000]
plt.xticks(xaxis)
plt.show()