import librosa
import numpy as np
import os
from sklearn.mixture import GaussianMixture
import pyaudio
import argparse
import wave
import sys

#when run the code with --train, it will record the user's voice for multiple times and get user name, save the voice as .wav file and train a model to recognize the user's voice with mfcc and gmm
#when run the code with --test, it will recognize the user's voice with the trained model

parser = argparse.ArgumentParser(description='Train or test the voice recognition model')
parser.add_argument('--train', action='store_true', help='train the voice recognition model')
parser.add_argument('--test', action='store_true', help='test the voice recognition model')
args = parser.parse_args()

# record the user's voice for multiple times and get user name, save the voice as .wav file and train a model to recognize the user's voice with mfcc and gmm

def prepare_train_data_dir(user_name):
    if not os.path.exists("train_data"):
        os.mkdir("train_data")
    # create a directory to store the training data
    if not os.path.exists("train_data" + os.sep + user_name):
        os.mkdir("train_data" + os.sep + user_name)

def record_voice(user_name):
    print("* recording")
    try:
        # Use pyaudio to record the voice
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        frames = []
    except Exception as e:
        print(str(e))
        exit(1)
    # record the voice for 5 seconds
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        WAVE_OUTPUT_FILENAME = "train_data" + os.sep + user_name + os.sep + user_name + "_" + str(i) + ".wav"
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def train_voice():
    # record the user's voice for multiple times
    # get user's name
    user_name = input("Please input your name: ")
    prepare_train_data_dir(user_name)
    # record the user's voice for 5 times, after pressing enter, the user should say the sentence "the quick brown fox jumps over the lazy dog" within 5 seconds
    # use 
    try:
        for i in range(5):
            print("Please say the sentence 'the quick brown fox jumps over the lazy dog' after pressing enter")
            input()
            record_voice(user_name=user_name)
            print("Recording " + str(i + 1) + " is done")
        print("Recording is done") 
    except:
        the_exception = sys.exc_info()[0]
        print("An exception occurred: ", the_exception)
        exit(1)
    # train a model to recognize the user's voice with mfcc and gmm
    # load the voice data
    voice_data = []
    for i in range(5):
        voice_data.append(librosa.load("train_data" + os.sep + user_name + os.sep + user_name + "_" + str(i) + ".wav", sr=16000)[0])
    # extract mfcc features
    mfcc_features = []
    for i in range(5):
        mfcc_features.append(librosa.feature.mfcc(y=voice_data[i], sr=16000, n_mfcc=13)) # 13 mfcc features, sr may be lower for SONY Spresence board?
    # train a gmm model
    gmm = GaussianMixture(n_components=5)
    gmm.fit(np.concatenate(mfcc_features, axis=1).T)
    # save the model
    np.save("train_data" + os.sep + user_name + os.sep + "gmm.npy", gmm)
    print("Model saved successfully")
    
def recognize_voice(user_name):
    # recognize the user's voice
    # load the model
    gmm = np.load("train_data" + os.sep + user_name + os.sep + "gmm.npy")
    # record the user's voice
    print("Say the sentence 'the quick brown fox jumps over the lazy dog' after pressing enter")
    os.system("arecord -D plughw:1,0 -f S16_LE -r 16000 -d 5 test.wav")
    # load the voice data
    voice_data = librosa.load("test.wav", sr=16000)[0]
    # extract mfcc features
    mfcc_features = librosa.feature.mfcc(y=voice_data, sr=16000, n_mfcc=13)
    # recognize the voice
    score = gmm.score(mfcc_features.T)
    print("The score is: ", score)
    if score > -100:
        print("The voice is recognized as the user" + user_name + " with score: ", score)
    else:
        print("The voice is not recognized as the user" + user_name + " with score: ", score)

if args.train:
    train_voice()
elif args.test:
    user_name = input("Please input your name: ")
    recognize_voice(user_name=user_name)
else:
    print("Please specify --train or --test")
    exit(1)
