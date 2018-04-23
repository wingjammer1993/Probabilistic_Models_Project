import numpy as np
import matplotlib.pyplot as plt


def generate_time_series(alpha_1, alpha_2, sigma):
	samples = 100
	mu_y1 = np.random.normal(0, sigma)
	mu_y2 = np.random.normal(0, sigma)
	mu_z = np.random.normal(0, sigma)
	y_1 = [mu_y1]*100
	y_2 = [mu_y2]*100
	z = [alpha_1*y_1[0] + alpha_2*y_2[0] + mu_z]*100
	for i in range(1, samples):
		y_2[i] = y_1[i-1] + mu_y2
		y_1[i] = alpha_1*y_1[i-1] + alpha_2*y_2[i-1] + mu_y1
		z[i] = y_1[i] + mu_z
	plt.plot(z)
	plt.show()


generate_time_series(1.5, -1, 3)