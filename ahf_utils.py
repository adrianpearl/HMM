import numpy as np
import math
from matplotlib import pyplot as plt

def mvGaussian(x, mu, sigma):
	p = len(x)
	k = 1 / (math.pow(2*math.pi, p*0.5)*math.pow(np.linalg.det(sigma), 0.5))
	e = math.exp(-0.5*(x-mu).T.dot(np.linalg.inv(sigma)).dot(x-mu))
	return k*e
	