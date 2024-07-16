#!/usr/bin/env python3
import argparse
import os
import glob
import sys
from typing import Optional
from voice_auth import voice_auth
from voice_auth import voice_record
import logging
import utilities as ut

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

THRESHOLD = -300
SECONDS = 4
BASEPATH = os.path.dirname(__file__)
logging.debug('BASEPATH: %s', BASEPATH)
logging.debug('audio folder: %s', os.path.join(BASEPATH, f'../audio'))
NUM_SAMPLE = 6
phrase = 'The quick fox jumps nightly above the wizard'

ut.check_folder(os.path.join(BASEPATH, f'../audio'))
ut.check_folder(os.path.join(BASEPATH, f'../audio_models'))

def authenticate():
    # f = open(os.path.join(BASEPATH, 'threshold.txt'), 'r')
    # THRESHOLD = float(f.read())
    # f.close()
    #
    # path = os.path.join(BASEPATH, '../audio/compare.wav')
    # model, prob = voice_auth.compare(voice_record.record(path, SECONDS))
    # logging.debug(f"{model}, {prob}")
    #
    # if prob and prob > THRESHOLD:
    #     print('Verified')
    #     return True
    # else:
    #     return False

    # read multiple users
    score = []
    model_path = os.path.join(BASEPATH, '/../audio_models')
    path = os.path.join(BASEPATH, '../audio/compare.wav')

    for files in model_path:
        username = files.split('/')[-1]
        logging.info(username)

        f_threshold = open(os.path.join(files, f'{username}/threshold.txt'), 'r')
        THRESHOLD = float(f_threshold.read())
        f_threshold.close()
        logging.info(THRESHOLD)
        model, prob = voice_auth.compare(voice_record.record(path, SECONDS))
        logging.debug(f"{model}, {prob}")

        if prob > THRESHOLD:
            score.append((username, prob))
            logging.info(f"{username}: {prob}, over {THRESHOLD}")
            return True
        else:
            score.append((username, 0))
            logging.info(f"{username}: {prob}, under {THRESHOLD}")

    if len(score) == 0: return False







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--auth', action='store_true', required=False)
    parser.add_argument('-p', '--phrase', required=False)
    args = parser.parse_args()



    if not args.auth:
        # files = glob.glob(os.path.join(BASEPATH, '../audio_models/*'))
        # for f in files:
        #     os.remove(f)


        dest = os.path.join(BASEPATH, f'../audio')
        phrase = args.phrase if args.phrase else phrase
        username = input('Please input your username: ')
        ut.check_folder(os.path.join(BASEPATH, f'../audio_models/{username}'))

        paths_modelling = []
        print("Please say the phrase:", phrase)
        if os.path.exists(os.path.join(dest, f'{username}1.wav')):
            for i in range (1, NUM_SAMPLE//2 + 1):
                path = os.path.join(dest, f'../audio/{username}1.wav')

                paths_modelling.append(path)
        else:
            for i in range(1, NUM_SAMPLE//2 + 1):
                promp = input('Press enter to record... ')

                path = os.path.join(dest, username + str(i) + '.wav')
                voice_record.record(path, SECONDS)
                paths_modelling.append(path)
        # logging.info(f"Path Modeling: $s", paths_modelling )
        print(f'Path Modeling: $s', paths_modelling)

        paths_training = []
        print("Please say the phrase:", phrase)
        if os.path.exists(os.path.join(dest, f'{username}{NUM_SAMPLE//2 + 2}.wav')):
            for i in range(4, int(NUM_SAMPLE) + 1):
                path = os.path.join(dest, f'{username}{i}.wav')
                paths_training.append(path)
        else:
            for i in range(4, int(NUM_SAMPLE) + 1):
                promp = input('Press enter to record... ')
                path = os.path.join(dest, username + str(i) + '.wav')
                voice_record.record(path, SECONDS)
                paths_training.append(path)
        # logging.info(f"Path Training: $s", paths_training)
        print(f'Path Training: $s', paths_training)

        voice_auth.build_model(username, paths_modelling)

        thresholds = []
        for path in paths_training:
            model, prob = voice_auth.compare(username,path)
            print(f"{model}, {prob}")
            thresholds.append(prob)

        THRESHOLD = (sum(thresholds) / len(thresholds)) - 0.5
        logging.debug(THRESHOLD)

        f = open(os.path.join(BASEPATH, f'../audio_models/{username}/threshold.txt'), 'w')
        f.write(str(THRESHOLD))
        f.close()

    else:
        sys.exit(1 if authenticate() else 0)