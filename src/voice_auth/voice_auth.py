import librosa 
import numpy as np
import os
import logging
import sklearn
import scipy.io.wavfile as wav
import sklearn.mixture
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

BASEPATH = os.path.dirname(__file__)

n_mfcc = 19



def get_mfcc(file):
    logging.debug(f'Loading {file}')
    y, sr = librosa.load(file, sr=None)
    logging.debug(f'Getting MFCC features for {file}, sr: {sr}')
    # Get the MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def train_gmm(user_name, paths):
    logging.info(f'Training GMM for {user_name}')
    dest = os.path.join(BASEPATH, f'../audio_models')
    combined_mfcc = np.asarray([])

    for path in paths:
        logging.debug(f'Loading {path}')
        mfcc = get_mfcc(path)
        logging.debug(f'MFCC shape for {path}: {mfcc.shape}')
        if combined_mfcc.size == 0:
            combined_mfcc = mfcc
        else:
            combined_mfcc = np.concatenate((combined_mfcc, mfcc), axis=1)

    if combined_mfcc.size != 0:
        logging.debug(f'# Samples: {len(paths)}')
        logging.debug(f'MFCC shape: {combined_mfcc.shape}')
        
        # Preprocess the combined MFCC
        scaler = RobustScaler()
        preprocessed_mfcc = scaler.fit_transform(combined_mfcc.T).T
        
        gmm = sklearn.mixture.GaussianMixture(n_components=1, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(preprocessed_mfcc.T)
        
        # Save both the GMM model and the scaler
        model_path = os.path.join(BASEPATH, f'../../audio_models/{user_name}_model.joblib')
        scaler_path = os.path.join(BASEPATH, f'../../audio_models/{user_name}_scaler.joblib')
        
        joblib.dump(gmm, model_path)
        joblib.dump(scaler, scaler_path)
        
        return True
    else:
        logging.warning(f'No sample features found for {user_name}')
        return False

def recognize_voice(user_name, mfcc):
    model_path = os.path.join(BASEPATH, f'../../audio_models/{user_name}_model.joblib')
    scaler_path = os.path.join(BASEPATH, f'../../audio_models/{user_name}_scaler.joblib')
    
    gmm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Preprocess the input MFCC
    mfcc_preprocessed = scaler.transform(mfcc.T).T
    
    score = gmm.score(mfcc_preprocessed.T)
    return score
    # model_path = os.path.join(BASEPATH, f'../../audio_models/{user_name}_model.joblib')
    # scaler_path = os.path.join(BASEPATH, f'../../audio_models/{user_name}_scaler.joblib')
    
    # gmm = joblib.load(model_path)
    # scaler = joblib.load(scaler_path)
    
    # # Preprocess the input MFCC
    # mfcc_preprocessed = scaler.transform(mfcc.T)
    
    # score = gmm.score(mfcc_preprocessed)
    # return score

def compare(spath):
    """ Compares audio features against all models to find closest match above given threshold
    Parameters:
    spath: str              - path of WAV file to compare
    """
    models_src = os.path.join(BASEPATH, '../../audio_models')
    model_paths = [os.path.join(models_src, fname) for fname in
        os.listdir(models_src) if fname.endswith('_model.joblib')]

    best_probability = None
    debug_every_model = []

    features = get_mfcc(spath)
    logging.debug(f'Features shape: {features.shape}')

    for model_path in model_paths:
        user_name = os.path.splitext(os.path.basename(model_path))[0].replace('_model', '')
        scaler_path = os.path.join(models_src, f'{user_name}_scaler.joblib')

        if not os.path.exists(scaler_path):
            logging.warning(f"Scaler not found for {user_name}, skipping...")
            continue

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Preprocess the features using the scaler
        preprocessed_features = scaler.transform(features.T).T

        ll = np.array(model.score(preprocessed_features.T)).sum()
        logging.info(f'Log likelihood for {user_name}: {ll}')

        if best_probability is None or ll > best_probability:
            best_probability = ll

        debug_every_model.append((user_name, ll))

    logging.debug(debug_every_model)
    logging.info(f'Best probability: {best_probability}')
    return best_probability