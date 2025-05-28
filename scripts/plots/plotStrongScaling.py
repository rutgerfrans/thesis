import matplotlib.pyplot as plt
workers = [1,2,4,8,12,16]
tf_times = [
161.980434,
86.609584,
62.0831,
57.875627,
61.779232,
52.640887
]
pt_times = [
134.739212,
71.838127,
52.458581,
40.815267,
52.849716,
49.273728
]
workers_sam = [1,2,4,8,12,16,32,64]
sam_times = [ 
201.866583,
132.347872,
102.868706,
92.060057,
92.548227,
93.74776,
94.657777,
97.051366
]

plt.figure()
plt.plot(workers, tf_times, marker='o', label='TensorFlow')
plt.plot(workers, pt_times, marker='o', label='PyTorch')
plt.plot(workers_sam, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Workers')
plt.ylabel('Wall-clock Time (s)')
plt.title('Strong Scaling: Number of Workers vs Wall-clock Time (s)')
plt.legend()
plt.xticks(workers_sam)
plt.show()
