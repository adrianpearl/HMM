import numpy as np
from HMM import MultinomialHMM, GaussianMixtureHMM
import matplotlib.pyplot as plt
from ahf_utils import mvGaussian

testword = "on"
words = ["yes", "no", "up", "down", "on", "off"]

np.set_printoptions(precision=5, linewidth=120, suppress=False)

yes_hmm = GaussianMixtureHMM.loadModel("yes")
no_hmm = GaussianMixtureHMM.loadModel("no")
on_hmm = GaussianMixtureHMM.loadModel("on")
off_hmm = GaussianMixtureHMM.loadModel("off")
up_hmm = GaussianMixtureHMM.loadModel("up")
down_hmm = GaussianMixtureHMM.loadModel("down")

#hmm.printmodel()

#logps=[]
for word in ["on"]:

    """
    wordlogps = []
    print("testing " + word)
    test = np.loadtxt("X_test/" + word + "/X.csv", delimiter=",")
    sequence_lengths = np.loadtxt("X_test/" + word + "/S.csv", delimiter="\n").astype(int)

    t = np.arange(0, sequence_lengths.shape[0], 1)
    start = 0
    for i, length in enumerate(sequence_lengths):
        if i%20 == 0:
            print(i)

        utterance = test[start : start+length]

        logps = [yes_hmm.logpsequence(utterance), no_hmm.logpsequence(utterance), up_hmm.logpsequence(utterance), down_hmm.logpsequence(utterance), on_hmm.logpsequence(utterance), off_hmm.logpsequence(utterance)]
        #print([int(i) for i in logps])
        wordlogps.append(logps)
        start = start+length

    result = np.array(wordlogps)
    np.savetxt("testing/%s.csv" % (word), result, delimiter=",", fmt="%1.5g")
    """

    result = np.loadtxt("testing/%s.csv" % (word), delimiter=",")

    plt.figure(1)
    plt.plot(result[:,0], 'r-')
    plt.plot(result[:,1], 'r-')
    plt.plot(result[:,2], 'r-')
    plt.plot(result[:,3], 'r-')
    plt.plot(result[:,4], 'b-')
    plt.plot(result[:,5], 'r-')

    plt.show()

    break

"""
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
