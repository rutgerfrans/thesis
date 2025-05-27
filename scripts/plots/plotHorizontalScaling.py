import matplotlib.pyplot as plt
neurons = [16,32,64,128,256]
tf_times = [150.975737,157.270153,161.73564,180.534572,246.724747]
pt_times = [27.429125,49.66625,84.425434,251.323491,1531.089419]
sam_times = [126.902194,186.677764,176.306397,397.822766,1779.9464]

plt.figure()
plt.plot(neurons, tf_times, marker='o', label='TensorFlow')
plt.plot(neurons, pt_times, marker='o', label='PyTorch')
plt.plot(neurons, sam_times, marker='o', label='SAM')

plt.xlabel('Number of Neurons in a Hidden-Layer')
plt.ylabel('Wall-clock Time (s)')
plt.title('Horizontal Scaling: Scaling Model Width vs Wall-clock Time (s)')
plt.legend()
plt.xticks(neurons)
plt.show()