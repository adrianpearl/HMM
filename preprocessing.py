import os
import numpy as np
from python_speech_features import mfcc
import wave

words = ["yes", "no", "up", "down", "on", "off"]

def preprocess(word):

	directory = os.fsencode("speech_commands_v0.01/" + word)

	i = 0
	X_train = []
	X_train_sl = []
	
	X_test = []
	X_test_sl = []
	for file in os.listdir(directory):
	
		if i%100 == 0:
			print(i)
	
		filename = os.fsdecode(file)
		if filename.endswith(".wav"):
	
			filepath = "speech_commands_v0.01/" + word + "/" + filename
		
			stdwav = wave.open(filepath)
			stdrate = stdwav.getframerate()
			stdsig = np.frombuffer(stdwav.readframes(stdwav.getnframes()), np.int16)

			std_mfcc_feat = mfcc(stdsig,stdrate)
		
			mfcclist = std_mfcc_feat.tolist()
			
			if np.random.randint(10) > 0:
				X_train = X_train + mfcclist
				X_train_sl.append(std_mfcc_feat.shape[0])
			else:
				X_test = X_test + mfcclist
				X_test_sl.append(std_mfcc_feat.shape[0])
	
		i = i + 1

	X_train_np = np.array(X_train)
	X_test_np = np.array(X_test)
	
	print("saving ", X_train_np.shape[0], " training sequences")
	np.savetxt("X_train/" + word + "/X.csv", X_train_np, delimiter=",", fmt="%1.5g")
	np.savetxt("X_train/" + word + "/S.csv", np.int_(X_train_sl), delimiter="\n", fmt="%1.5g")
	
	print("saving ", X_test_np.shape[0], " testing sequences")
	np.savetxt("X_test/" + word + "/X.csv", X_test_np, delimiter=",", fmt="%1.5g")
	np.savetxt("X_test/" + word + "/S.csv", np.int_(X_test_sl), delimiter="\n", fmt="%1.5g")

def savesubset(word, number_of_sequences):
	assert number_of_sequences > 1
	X_train_np = np.loadtxt("X_train/" + word + "/X.csv", delimiter=",")
	sequence_lengths = np.loadtxt("X_train/" + word + "/S.csv", delimiter="\n")
	sl = sequence_lengths[0:number_of_sequences].astype(int)
	sx = X_train_np[0:np.sum(sl)]

	np.savetxt("X_train/" + word + "/miniX.csv", sx, delimiter=",", fmt="%1.5g")
	np.savetxt("X_train/" + word + "/miniS.csv", sl, delimiter="\n", fmt="%1.5g")
	
for word in words:
	print(word)
	preprocess(word)