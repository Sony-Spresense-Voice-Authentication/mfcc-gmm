import argparse
import os
import glob
import sys
import utilities
import voice_auth.voice_record as voice_record
import voice_auth.voice_auth as voice_auth
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

THRESHOLD = 0.5
SECONDS = 5
BASEPATH = os.path.dirname(__file__)
NUM_SAMPLE = 6
phrase = 'The quick fox jumps nightly above the wizard'

# Check if the audio directory exists
utilities.check_folder(os.path.join(BASEPATH, f'../audio'))
utilities.check_folder(os.path.join(BASEPATH, f'../audio_models'))

# functions to record the voice and authenticate the user
def authenticate():
    # Authenticate the user
    # Get the voice recording and save to "sample.wav"
    dest = os.path.join(BASEPATH, f'../audio/sample.wav')
    print("Recording the voice for authentication..")
    
    # Record the voice for 5 seconds
    voice_record.record(dest, SECONDS)

    # Get the MFCC features for the authentication
    # mfcc = voice_auth.get_mfcc(dest)
    
    # # Used the trained GMM model to recognize the user's voice
    # score = voice_auth.recognize_voice(mfcc)
    # logging.info(f'Score: {score}')


    # Get the MFCC features for the authentication
    mfcc = voice_auth.get_mfcc(dest)
    scores = []
    for file in glob.glob(os.path.join(BASEPATH, '../audio_models/*')):
        logging.info(f'Loading {file}')
        user = os.path.basename(file).split('.')[0]
        logging.info(f'Checking {user}')
        score = voice_auth.recognize_voice(user,mfcc)
        scores.append((user, score))
        logging.info(f'Score for {user}: {score}')

    # Get the user with the highest score
    # if scores.count == 0:
    #     utilities.break_and_signal('No scores found for the users')
    # user, score = max(scores, key=lambda x: x[1])
    # logging.info(f'User: {user}, Score: {score}')
    # if score > THRESHOLD:
    #     print(f'User {user} authenticated')
    # else:
    #     print('User not authenticated')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voice authentication')
    parser.add_argument('-a','--auth', action='store_true', help='Authenticate the user',required=False)
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Threshold for voice authentication',required=False)
    parser.add_argument('-s','--seconds', type=int, default=SECONDS, help='Seconds for voice recording',required=False)
    parser.add_argument('-p','--phrase', type=str, default=phrase, help='Phrase for voice recording',required=False)
    args = parser.parse_args()
    
    if not args.auth:
        # Remove all the files in the audio_models directory
        remove_models = False
        if remove_models:
            files = glob.glob(os.path.join(BASEPATH, '../audio_models/*'))
            if not files:
                logging.info('No files found in the audio_models directory')
            else:
                for f in files:
                    logging.info(f'Removing {f}')
                    os.remove(f)
            

        # Remove all the files in the audio directory
        remove_audio = False
        if remove_audio:
            files = glob.glob(os.path.join(BASEPATH, '../audio/*'))
            for f in files:
                logging.info(f'Removing {f}')
                os.remove(f)
            

        dest = os.path.join(BASEPATH, f'../audio')
        phrase = args.phrase if args.phrase else phrase
        username = input('Please input your username: ')
        # utilities.check_folder(username)
        
        paths_modeling = []
        path_training = []
        if not os.path.exists(os.path.join(dest, f'{username}_0.wav')):
            utilities.break_and_signal(f'No samples found for {username}')
            print("Recording the voice for " + username + "..")
            for i in range(0, int(NUM_SAMPLE // 2) + 1):
                logging.info(f'Recording {i + 1} of {NUM_SAMPLE}')
                print("Please say the following phrase: ", phrase)
                prompt = input("Press enter to start recording.")
                path = os.path.join(dest, f'{username}_{i}.wav')
                logging.debug(f'Saving to {path}')
                voice_record.record(path, args.seconds)
                paths_modeling.append(path)

            print("Recording the voice for " + username + "..")
            for i in range(int(NUM_SAMPLE // 2) + 1, NUM_SAMPLE):
                logging.info(f'Recording {i + 1} of {NUM_SAMPLE}')
                print("Please say the following phrase: ", phrase)
                prompt = input("Press enter to start recording.")
                path = os.path.join(dest, f'{username}_{i}.wav')
                logging.debug(f'Saving to {path}')
                voice_record.record(path, args.seconds)
                path_training.append(path)
        else:
            # Count the number of user's samples
            i = 0
            while os.path.exists(os.path.join(dest, f'{username}_{i}.wav')):
                i += 1
            logging.info(f'Found {i} samples for {username}')
            for j in range(0, int(NUM_SAMPLE // 2)):
                path = os.path.join(dest, f'{username}_{j}.wav')
                logging.debug(f'Loading {path} to modeling set')
                paths_modeling.append(path)
            logging.debug(f'Paths modeling: {paths_modeling}')
            for j in range(int(NUM_SAMPLE // 2) + 1, i):
                path = os.path.join(dest, f'{username}_{j}.wav')
                logging.debug(f'Loading {path} to training set')
                path_training.append(path)
            logging.debug(f'Paths training: {path_training}')


        # For all modeling paths, get the MFCC features and train the GMM model
        voice_auth.train_gmm(username, paths_modeling)

        # Get the threshold for the user from the training set
        logging.info('Getting the threshold for the user')
        thresholds = []
        for path in path_training:
            prob = voice_auth.compare(path)
            thresholds.append(prob)
            logging.info(f'Probability for {path}: {prob}')

        THRESHOLD = (sum(thresholds) / len(thresholds)) - 0.5
        logging.debug(THRESHOLD)

        f = open(os.path.join(BASEPATH, 'threshold.txt'), 'w')
        f.write(str(THRESHOLD))
        f.close()

    else:
        sys.exit(1 if authenticate() else 0)
