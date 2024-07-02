import librosa 
import numpy as np
import os
import logging
import sklearn
import scipy.io.wavfile as wav
import sklearn.mixture
import joblib

BASEPATH = os.path.dirname(__file__)

def get_mfcc(file):
    logging.debug(f'Loading {file}')
    y, sr = librosa.load(file, sr=None)
    logging.debug(f'Getting MFCC features, sr: {sr}')
    # Get the MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = sklearn.preprocessing.scale(mfcc, axis=1)
    return mfccs

def train_gmm(user_name,paths):
    logging.info(f'Training GMM for {user_name}')
    # # Train a GMM model and save the model to a file
    # gmm = sklearn.mixture.GaussianMixture(n_components=5)
    # gmm.fit(np.concatenate(mfcc, axis=1).T)
    # np.save(os.path.join(BASEPATH, f'../audio_models/{user_name}.npy'), gmm)
    # return gmm
    # train a model to recognize the user's voice with mfcc and gmm
    dest = os.path.join(BASEPATH, f'../audio_models')
    combined_mfcc = np.asarray([])
    logging.debug(f'Paths: {paths}')

    for path in paths:
        logging.debug(f'Loading {path}')
        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        logging.debug(f'MFCC shape for {path}: {mfcc.shape}')
        if combined_mfcc.size == 0:
            combined_mfcc = mfcc
        else:
            combined_mfcc = np.concatenate((combined_mfcc, mfcc), axis=1)

    if combined_mfcc.size != 0:
        logging.debug(f'# Samples: {len(paths)}')
        logging.debug(f'MFCC shape: {combined_mfcc.shape}')
        gmm = sklearn.mixture.GaussianMixture(n_components=1, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(combined_mfcc.T)
        joblib.dump(gmm, os.path.join(BASEPATH, f'../../audio_models/{user_name}.joblib'))
        # np.save(os.path.join(BASEPATH, f'../../audio_models/{user_name}.npy'), gmm)
        return True
    else:
        logging.warning(f'No sample features found for {user_name}')
        return False
    

def recognize_voice(user_name,mfcc):
    gmm = joblib.load(os.path.join(BASEPATH, f'../../audio_models/{user_name}.joblib'))
    score = gmm.score(mfcc.T)
    return score

def compare(spath):
    """ Compares audio features against all models to find closest match above given threshold
    Parameters:
    spath: str              - path of WAV file to compare
    """
    models_src = os.path.join(BASEPATH, '../../audio_models')
    model_paths = [os.path.join(models_src, fname) for fname in
        os.listdir(models_src) if fname.endswith('.joblib')]

    # sampling_rate, data = wav.read(path)

    best_model = None
    best_probability = None
    debug_every_model = []
    for path in model_paths:
        # model_name = os.path.splitexlt(os.path.basename(path))[0]
        model = joblib.load(path)
        features = get_mfcc(spath)
        ll = np.array(model.score(features.T)).sum()

        # if best_model is None:
        #     best_model = model_name
        if best_probability is None:
            best_probability = ll
        elif ll > best_probability:
            # best_model = model_name
            best_probability = ll
        # debug_every_model.append((model_name, ll))

    logging.debug(debug_every_model)
    # return best_model, best_probability
    logging.info(f'Best probability: {best_probability}')
    return best_probability