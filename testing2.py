import numpy as np
from HMM import MultinomialHMM, GaussianMixtureHMM
import matplotlib.pyplot as plt
import pydub
import pyaudio
import wave
from python_speech_features import mfcc

yes_hmm = GaussianMixtureHMM.loadModel("yes")
no_hmm = GaussianMixtureHMM.loadModel("no")
on_hmm = GaussianMixtureHMM.loadModel("on")
off_hmm = GaussianMixtureHMM.loadModel("off")
up_hmm = GaussianMixtureHMM.loadModel("up")
down_hmm = GaussianMixtureHMM.loadModel("down")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "output"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME + ".wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

wfr = pydub.AudioSegment.from_wav(WAVE_OUTPUT_FILENAME + ".wav")
slice_len = 100

nonsilent = pydub.silence.split_on_silence(wfr, min_silence_len=400, silence_thresh=-32, keep_silence=300, seek_step=1)

if len(nonsilent) > 0:

    word = nonsilent[0]
    word.export(WAVE_OUTPUT_FILENAME + "_cut.wav", format="wav")

    stdwav = wave.open(WAVE_OUTPUT_FILENAME + "_cut.wav")
    stdrate = stdwav.getframerate()
    stdsig = np.frombuffer(stdwav.readframes(stdwav.getnframes()), np.int16)

    utterance = mfcc(stdsig,stdrate)
    #std_len = std_mfcc_feat.shape[0]

    logps = {"yes": yes_hmm.logpsequence(utterance), "no": no_hmm.logpsequence(utterance), "up": up_hmm.logpsequence(utterance), "down": down_hmm.logpsequence(utterance), "on": on_hmm.logpsequence(utterance), "off": off_hmm.logpsequence(utterance)}

    for key, value in logps.items():
        print(key, value)
"""
Plotting RMS
for chunk in nonsilent:
    wfrrms = []
    print(chunk.rms)
    #print(wfr.max_possible_amplitude)

    seg_len = len(chunk)
    last_slice_start = seg_len - slice_len
    slice_starts = range(0, last_slice_start + 1, 1)

    for i in slice_starts:
        audio_slice = chunk[i:i + slice_len]
        wfrrms.append(audio_slice.rms)

    #print(wfrrms)

    plt.figure(1)
    plt.plot(wfrrms, 'r-')


plt.show()"""

"""
Playback of cut version
p2 = pyaudio.PyAudio()

if len(nonsilent) > 0:
    word = nonsilent[0]
    word.export(WAVE_OUTPUT_FILENAME + "_cut.wav", format="wav")

    wfr = wave.open(WAVE_OUTPUT_FILENAME + "_cut.wav", 'rb')

    stream = p2.open(format=p2.get_format_from_width(wfr.getsampwidth()),
                    channels=wfr.getnchannels(),
                    rate=wfr.getframerate(),
                    output=True)

    data = wfr.readframes(CHUNK)

    print("read frames done")

    while data != b'':
        stream.write(data)
        data = wfr.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
else:
    print("no nonsilent chunks")

p2.terminate()"""
