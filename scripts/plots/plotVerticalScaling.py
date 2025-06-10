import matplotlib.pyplot as plt
hiddenlayers = [2,3,4,5]
tf_times = [
50474.8087687404,
42906.7997032405,
31324.0226830691,
27654.5329504767
]
pt_times = [
47026.277574929,
44235.1800753977,
33913.7924565892,
27628.6986066432
]
sam_times = [
53088.8668489293,
43402.7642842715,
33573.4421516113,
28816.7201482512
]

plt.figure()
plt.plot(hiddenlayers, tf_times, marker='o', label='TensorFlow')
plt.plot(hiddenlayers, pt_times, marker='o', label='PyTorch')
plt.plot(hiddenlayers, sam_times, marker='o', label='SAM')

plt.xlabel('# Hidden-Layers')
plt.ylabel('Images/s')
plt.title('Throughput vs. Model Depth')
plt.legend()
xaxis = [2,3,4,5]
plt.xticks(xaxis)
plt.ylim(bottom=0)
plt.savefig('plots/images/verticalscalingresult.png', dpi=300,bbox_inches='tight')
#plt.show()