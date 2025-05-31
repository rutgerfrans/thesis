import matplotlib.pyplot as plt
workers = [1,2,4,8,12]
tf_times = [
14457.5479225245,
27494.044016796,
40144.1563272326,
41137.4406348165,
39092.6173494404
]
pt_times = [
18037.9757110239,
32940.2585999707,
54336.1184744727,
60046.1484674046,
51123.8633434423
]
sam_times = [ 
18774.8966864373,
34616.3279103202,
55068.2487193248,
62159.4584357184,
50369.430389741
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
