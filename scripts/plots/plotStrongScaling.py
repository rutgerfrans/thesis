import matplotlib.pyplot as plt
workers = [1,2,4,8,12]
tf_times = [
17642.1951952677,
30880.2667790015,
50474.8087687404,
49179.8046239492,
45466.2926537723
]
pt_times = [
17318.3050931089,
30516.8056924145,
47026.277574929,
47871.7579265013,
45936.3211238815
]
sam_times = [ 
18677.0883022865,
31976.735111486,
53088.8668489293,
50826.0791315633,
47308.7033464267
]

plt.figure()
plt.plot(workers, tf_times, marker='o', label='TensorFlow')
plt.plot(workers, pt_times, marker='o', label='PyTorch')
plt.plot(workers, sam_times, marker='o', label='SAM')

plt.xlabel('# Workers')
plt.ylabel('Images/s')
plt.title('Throughput vs. Workers')
plt.legend()
xaxis = [0,1,2,4,8,12]
plt.xticks(xaxis)
plt.ylim(bottom=0)
plt.savefig('plots/images/strongscalingresult.png', dpi=300,bbox_inches='tight')
#plt.show()
