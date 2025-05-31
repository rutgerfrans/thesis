import matplotlib.pyplot as plt
hiddenlayers = [2,3,4,5]
tf_times = [
38393.3699257343,
31529.736502445,
26699.3659734936,
23331.1530046807
]
sam_times = [
48340.822132362,
39711.1569289615,
33803.3572790213,
29365.4313812459
]
pt_times = [
44995.9270249306,
38308.3810629158,
32676.0780760576,
28444.7465065822
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