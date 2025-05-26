import matplotlib.pyplot as plt
datasetsizes = [10000,30000,60000]
tf_times = [33.503849,79.777392,150.975737]
pt_times = [18.565649,24.168294,27.429125]
sam_times = [23.117687,72.903935,163.05613]

plt.figure()
plt.plot(datasetsizes, tf_times, marker='o', label='TensorFlow')
plt.plot(datasetsizes, pt_times, marker='o', label='PyTorch')
plt.plot(datasetsizes, sam_times, marker='o', label='SAM')

plt.xlabel('Dataset Size')
plt.ylabel('Wall-clock Time (s)')
plt.title('Weak Scaling: Scaling Dataset vs Wall-clock Time (s)')
plt.legend()
plt.xticks(datasetsizes)
plt.show()