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
		p_priori = np.dot(self.matrix_A, np.dot(p_init, self.matrix_A.T))
		return x_priori, p_priori

	def update(self, x_pri, p_pri, observation):
		innovation = observation - np.dot(self.matrix_H, x_pri)
		innovation_cov = self.matrix_R + np.dot(self.matrix_H, np.dot(p_pri, self.matrix_H.T))
		kalman_gain = np.dot(p_pri, np.dot(self.matrix_H.T, np.linalg.inv(innovation_cov)))
		x_post = x_pri + np.dot(kalman_gain, innovation)
		term = np.dot(kalman_gain, self.matrix_H)
		m = 1 - term
		p_post = np.dot(m, np.dot(m.T, p_pri)) + np.dot(kalman_gain, np.dot(self.matrix_R, kalman_gain.T))
		return x_post, p_post


if __name__ == '__main__':

	z = Part_2a.generate_time_series(1.5, -1, 3, 200)
	a = np.array([[1.5, -1], [1, 0]])  # state transition matrix
	p = np.array([[0.9, 0], [0, 0.9]])  # process covariance
	q = np.array([[1.732, 0], [0, 1.732]])  # state covariance
	h = np.array([1, 0])  # observation matrix
	r = np.array([[1.732, 0], [0, 1.732]])  # observation covariance
	kf = KalmanFilter(a, q, h, r)
	x_initial = [0, 0]
	p_initial = copy.deepcopy(p)
	z_pred = []
	for i in range(0, len(z)):
		x_in, p_in = kf.predict(x_initial, p_initial)
		x_postr, p_postr = kf.update(x_in, p_in, z[i])
		z_pred.append(np.dot(h, x_postr))
		x_initial = x_postr
		p_initial = p_postr
	print(z_pred)
	plt.plot(z, color='green')
	plt.plot(z_pred)
	plt.show()




