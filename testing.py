import numpy as np
from HMM import MultinomialHMM, GaussianMixtureHMM
import matplotlib.pyplot as plt
from ahf_utils import mvGaussian

testword = "on"
words = ["yes", "no", "up", "down", "on", "off"]

np.set_printoptions(precision=5, linewidth=120, suppress=False)

yes_hmm = GaussianMixtureHMM.loadModel("yes")
on_hmm = GaussianMixtureHMM.loadModel("on")
up_hmm = GaussianMixtureHMM.loadModel("up")
#hmm.printmodel()

#logps=[]

print("testing " + testword)
test = np.loadtxt("X_test/" + testword + "/X.csv", delimiter=",")
sequence_lengths = np.loadtxt("X_test/" + testword + "/S.csv", delimiter="\n").astype(int)
#sequence_lengths = sequence_lengths[0:200]
#yes_train = on_train[0:np.sum(sequence_lengths)]

yeslogps = yes_hmm.logps(test, sequence_lengths)
uplogps = up_hmm.logps(test, sequence_lengths)
onlogps = on_hmm.logps(test, sequence_lengths)

print("yes:")
print(yeslogps)
print("up:")
print(uplogps)
print("on:")
print(onlogps)

#logps.append(yeslogps)



"""
X_yes = np.loadtxt("X_train/yes/X.csv", delimiter=",")
sequence_lengths = np.loadtxt("X_train/yes/S.csv", delimiter="\n").astype(int)

start = np.sum(sequence_lengths[0:100].astype(int))
lengths = sequence_lengths[100:200].astype(int)
end = start + np.sum(lengths)
sx = X_yes[start:end]
wordlogps = hmm.logps(sx, lengths)
print(testword + " - testing:")
print(wordlogps)
"""

"""
obs = np.array([19.883, -9.5539, -16.605, 36.871, -21.879, -58.377, -9.9939, -26.14, -20.481, 25.459, -25.944, 12.437, 13.862])
mean = np.array([18.60888, -37.51102, 17.40778, -11.99703, -11.4298, 25.80397, -36.96028, 25.07043, -15.80151, 12.18687, -6.33444, -0.96385, -2.92354])
cov = np.array([1.06785, 27.80559, 12.07332, 15.17684, 22.93452, 23.46364, 41.6248, 33.05901, 56.25807, 69.85563, 91.65027, 34.82063, 42.54343])

print(mvGaussian(obs, mean, np.diag(cov)))

hmm = MultinomialHMM(2, 2)

hmm.A = np.array([[0.9, 0.1], [0.1, 0.9]])
hmm.B = np.array([[0.9, 0.1], [0.1, 0.9]])
hmm.pi = np.array([0.5, 0.5])

seq = np.random.randint(2, size=50)

print("sequence:")
print(seq)

a, b, c = hmm.forward_backward(seq, False)

print("alpha:")
print(a)
print("beta:")
print(b)
print("c's:")
print(c)
"""

"""
x=np.ones((3, 3))
y = np.full((3, 3), 2)

z = x[2,:] + y[:,1]

print(z)

x = np.arange(0., 1.01, 0.01)
y = np.sqrt(10000*x*(1-x))
plt.plot(x, y)
plt.grid(which='both')
plt.xlabel('size of partitions (% of total space)')
plt.ylabel('std deviation for # of people in partition for T=10,000')
plt.show()

x1 = np.arange(12.0).reshape((4, 3))
x2 = np.arange(3.0)

print(x1)
print(x1.shape)
print(x2)
print(x2.shape)

print(np.multiply(x1, x2)

x = np.array([1, 2, 3, 4, 5])
y = np.tile(x, (3, 1))
print(y)

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 1, 1, 2, 7])
z = np.array([1, 2, 5, 2, 4])

print(np.multiply(np.multiply(x, y), z))

gamma = np.ones((4, 2, 2))
print(np.sum(gamma, axis=0))
"""