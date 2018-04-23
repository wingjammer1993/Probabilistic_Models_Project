import numpy as np
import matplotlib.pyplot as plt


def generate_lds(alpha_1, alpha_2, sigma, samples):
	mu_y1 = np.random.normal(0, sigma)
	mu_y2 = np.random.normal(0, sigma)
	mu_z = np.random.normal(0, sigma)
	y_1 = [mu_y1]*samples
	y_2 = [mu_y2]*samples
	z = [alpha_1[0]*y_1[0] + alpha_2[0]*y_2[0] + mu_z]*samples
	alpha_mode_1 = alpha_1[0]
	alpha_mode_2 = alpha_2[0]
	mode = 0
	mode_record = [0]*samples
	for i in range(1, samples):
		if np.random.rand() < 0.3:
			draw_mode = np.random.multinomial(1, [1 / 3] * 3, size=1)
			mode = int(np.argmax(draw_mode))
			alpha_mode_1 = alpha_1[np.argmax(draw_mode)]
			alpha_mode_2 = alpha_2[np.argmax(draw_mode)]
		y_2[i] = y_1[i-1] + mu_y2
		y_1[i] = alpha_mode_1*y_1[i-1] + alpha_mode_2*y_2[i-1] + mu_y1
		z[i] = y_1[i] + mu_z
		mode_record[i] = mode
	plt.plot(z)
	plt.xlabel("Time")
	plt.ylabel("Observation")
	plt.title("Switched Linear Dynamic System with 3 modes")
	plt.show()
	return mode_record


def get_conditionals(modes):
	print("TODO")


record = generate_lds([1, 0.7, -0.75], [-0.5, -0.2, 0.3], 2, 200)
get_conditionals(record)

