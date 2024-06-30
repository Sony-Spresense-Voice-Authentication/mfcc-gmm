import argparse
import os
import glob
import sys
import utilities
from voice_auth import voice_record
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

THRESHOLD = 0.5
SECONDS = 5
BASEPATH = os.path.dirname(__file__)
NUM_SAMPLE = 2
phrase = 'The quick fox jumps nightly above the wizard'

# Check if the audio directory exists
utilities.check_folder(os.path.join(BASEPATH, f'../audio'))
utilities.check_folder(os.path.join(BASEPATH, f'../audio_models'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voice authentication')
    parser.add_argument('-a','--auth', action='store_true', help='Authenticate the user',required=False)
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Threshold for voice authentication',required=False)
    parser.add_argument('-s','--seconds', type=int, default=SECONDS, help='Seconds for voice recording',required=False)
    parser.add_argument('-p','--phrase', type=str, default=phrase, help='Phrase for voice recording',required=False)
    args = parser.parse_args()
    
    if not args.auth:
        # Remove all the files in the audio_models directory
        files = glob.glob(os.path.join(BASEPATH, '../audio_models/*'))
        for f in files:
            os.remove(f)
            logging.info(f'Removing {f}')

        # Remove all the files in the audio directory
        files = glob.glob(os.path.join(BASEPATH, '../audio/*'))
        for f in files:
            os.remove(f)
            logging.info(f'Removing {f}')

        dest = os.path.join(BASEPATH, f'../audio')
        phrase = args.phrase if args.phrase else phrase
        username = input('Please input your username: ')
        # utilities.check_folder(username)
        
        paths_modeling = []
        print("Recording the voice for " + username + "..")
        for i in range(1, int(NUM_SAMPLE // 2) + 1):
            logging.info(f'Recording {i + 1} of {NUM_SAMPLE}')
            print("Please say the following phrase: ", phrase)
            prompt = input("Press enter to start recording.")
            path = os.path.join(dest, f'{username}_{i}.wav')
            logging.debug(f'Saving to {path}')
            voice_record.record(path, args.seconds)
            paths_modeling.append(path)

        path_traning = []
        print("Recording the voice for " + username + "..")
        for i in range(int(NUM_SAMPLE // 2) + 1, NUM_SAMPLE + 1):
            logging.info(f'Recording {i + 1} of {NUM_SAMPLE}')
            print("Please say the following phrase: ", phrase)
            prompt = input("Press enter to start recording.")
            path = os.path.join(dest, f'{username}_{i}.wav')
            logging.debug(f'Saving to {path}')
            voice_record.record(path, args.seconds)
            paths_modeling.append(path)


    else:
        sys.exit(0)
