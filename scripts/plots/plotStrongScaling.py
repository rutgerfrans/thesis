import matplotlib.pyplot as plt
workers = [2, 4, 8, 12]
tf_times = [75.179126, 158.703419, 150.975737, 158.57327]
pt_times = [96.99536, 47.064342, 27.429125, 22.200962]
sam_times = [201.438586, 199.877597, 163.05613, 125.181624]

plt.figure()
plt.plot(workers, tf_times, marker='o', label='TensorFlow')
plt.plot(workers, pt_times, marker='o', label='PyTorch')
plt.plot(workers, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Workers')
plt.ylabel('Wall-clock Time (s)')
plt.title('Strong Scaling: Number of Workers vs Wall-clock Time (s)')
plt.legend()
plt.xticks(workers)
plt.show()
