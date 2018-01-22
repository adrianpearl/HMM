from HMM import GaussianMixtureHMM
import numpy as np

np.set_printoptions(precision=5, linewidth=120, suppress=True)

word = "up"

X_train = np.loadtxt("X_train/" + word + "/X.csv", delimiter=",")
sequence_lengths = np.loadtxt("X_train/" + word + "/S.csv", delimiter="\n").astype(int)

hmm = GaussianMixtureHMM(word, 5, "leftright", 1, True, 4)

hmm.estimate_state_distributions(X_train, sequence_lengths)
logps = hmm.train(X_train, sequence_lengths, 100)
#hmm.state_sequence_segmentation(X_train, sequence_lengths)

print("logps: ", logps)