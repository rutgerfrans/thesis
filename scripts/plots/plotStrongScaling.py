import matplotlib.pyplot as plt
workers = [1,2,4,8,12]
tf_times = [
14816.6043313602,
27710.5591455098,
38657.8634121041,
41468.2332512786,
38848.0063980077
]
pt_times = [
17812.1867003349,
33408.4434022062,
45750.3797138546,
58801.526399423,
45411.7861295603
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

plt.xlabel('Number of Workers')
plt.ylabel('Throughput img/s')
plt.title('Strong Scaling: Number of Workers vs Wall-clock Time (s)')
plt.legend()
xaxis = [0,1,2,4,8,12]
plt.xticks(xaxis)
plt.show()
