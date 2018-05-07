import numpy as np
import copy
import Part_2a
import matplotlib.pyplot as plt


class KalmanFilter:
	def __init__(self, A, Q, H, R,):
		self.matrix_A = copy.deepcopy(A)
		self.matrix_H = copy.deepcopy(H)
		self.matrix_Q = copy.deepcopy(Q)
		self.matrix_R = copy.deepcopy(R)

	def predict(self, x_init, p_init):
		x_priori = np.dot(self.matrix_A, x_init)
		p_priori = np.add(np.dot(self.matrix_A, np.dot(p_init, self.matrix_A.T)), self.matrix_Q)
		return x_priori, p_priori

	def update(self, x_pri, p_pri, observation):
		innovation = observation - np.dot(self.matrix_H, x_pri)
		innovation_cov = self.matrix_R + np.dot(self.matrix_H, np.dot(p_pri, self.matrix_H.T))
		kalman_gain = np.dot(p_pri, self.matrix_H) / innovation_cov
		x_post = x_pri + np.dot(kalman_gain, innovation)
		term = np.dot(kalman_gain, self.matrix_H)
		m = 1 - term
		p_post = np.dot(m, p_pri)
		return x_post, p_post


if __name__ == '__main__':

	z, y_true = Part_2a.generate_time_series(1.5, -1, 0.2, 200)
	a = np.array([[1.5, -1], [1, 0]])  # state transition matrix
	q = np.array([[0.04, 0], [0, 0.04]])  # state covariance
	h = np.array([1, 0])  # observation matrix
	r = np.array([0.2])  # observation covariance
	kf = KalmanFilter(a, q, h, r)
	x_initial = [0, 0]
	p = np.array([[50, 0], [0, 50]])  # process covariance
	p_initial = copy.deepcopy(p)
	x_posts = []
	for i in range(0, len(z)):
		x_in, p_in = kf.predict(x_initial, p_initial)
		x_postr, p_postr = kf.update(x_in, p_in, z[i])
		x_initial = x_postr
		p_initial = p_postr
		x_posts.append(x_postr[0])
	print(x_posts)

	plt.plot(z, color='green')
	plt.plot(y_true, color='red')
	plt.plot(x_posts, color='purple')
	plt.show()




