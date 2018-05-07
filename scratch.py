from filterpy.kalman import predict, update
from filterpy.kalman import KalmanFilter
import numpy as np
import copy
import Part_2a
import matplotlib.pyplot as plt
import math


def initialize_kf(x, P, F, Q, H, R):
	kft = KalmanFilter(dim_x=2, dim_z=1)
	kft.x = copy.deepcopy(x)
	kft.P = copy.deepcopy(P)
	kft.F = copy.deepcopy(F)
	kft.Q = copy.deepcopy(Q)
	kft.H = copy.deepcopy(H)
	kft.R = copy.deepcopy(R)
	return kft


if __name__ == '__main__':
	x = np.array([0, 0])
	P = np.array([[5, 0], [0, 5]])

	F = np.array([[0.5, 0.1], [1, 0]])
	Q = np.array([[0.04, 0], [0, 0.04]])

	H = np.array([[1, 0]])
	R = np.array([[0.04]])
	kf = initialize_kf(x, P, F, Q, H, R)
	zs, y_true = Part_2a.generate_time_series(0.5,  0.1, 0.2, 200)
	xs, cov = [], []
	for z in zs:
		kf.predict()
		kf.update(z)
		xs.append(kf.x)
		cov.append(kf.P)

	xs, cov = np.array(xs), np.array(cov)
	ys = [x[0] for x in xs]
	ps = []
	ys_top = []
	ys_bottom = []
	for idx, i in enumerate(ys):
		ys_top.append(i + math.sqrt(abs(cov[idx][0][0])))
		ys_bottom.append(i - math.sqrt(abs(cov[idx][0][0])))
	plt.plot(ys_top, linestyle=':', color='k', lw=2)
	plt.plot(ys_bottom, linestyle=':', color='k', lw=2)
	plt.plot(y_true, color="green", label="y1_true")
	plt.plot(ys, color="red",  label="prediction")
	plt.plot(zs, label="measurement")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Observations")
	plt.title("Kalman Filter predictions")
	plt.plot(ps)
	plt.show()



