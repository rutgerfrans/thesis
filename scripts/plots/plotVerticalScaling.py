import matplotlib.pyplot as plt
hiddenlayers = [2,3,4,5]
tf_times = [150.975737,151.200951,156.517822,163.499618]
pt_times = [27.429125,29.657657,39.121092,48.435512]
sam_times = [126.902194,132.180497,145.791522,160.472369]

plt.figure()
plt.plot(hiddenlayers, tf_times, marker='o', label='TensorFlow')
plt.plot(hiddenlayers, pt_times, marker='o', label='PyTorch')
plt.plot(hiddenlayers, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Hidden-Layers')
plt.ylabel('Wall-Clock Time (s)')
plt.title('Vertical Scaling: Scaling Model Depth vs Wall-clock Time (s)')
plt.legend()
plt.xticks(hiddenlayers)
plt.show()