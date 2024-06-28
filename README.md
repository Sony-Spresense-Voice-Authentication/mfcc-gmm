# Voice Recognition using MFCC and GMM

This Python script allows you to train and test a voice recognition model using Mel-Frequency Cepstral Coefficients (MFCC) and Gaussian Mixture Models (GMM).

## Requirements

The script requires the following Python libraries:

- `librosa`: A Python package for audio and music analysis
- `numpy`: A fundamental library for scientific computing in Python
- `scikit-learn`: A machine learning library for Python
- `pyaudio`: A Python binding for the PortAudio library, used for audio I/O

You can install these dependencies by running the following command when you use Windows:

```
pip install -r requirements.txt
```

Specifically for MacOS, you should install `portaudio` before installing `PyAudio`:

```zsh
brew install portaudio
pip install pyaudio
```

For more systems just refer to https://pypi.org/project/PyAudio/ .

## Usage

The script has two main modes of operation: `--train` and `--test`.

1. **Training the Model**:
   
   - Run the script with the `--train` argument:
     ```
     python mfcc_gmm_train_test.py --train
     ```
   - The script will prompt you to enter your name, and then record your voice for 5 seconds, 5 times.
   - The recorded voice samples will be saved in the `train_data` directory, with a subdirectory named after your username.
   - The script will then train a GMM model using the recorded voice samples and save the model in the `train_data` directory.
   
2. **Testing the Model**:
   - Run the script with the `--test` argument:
     ```
     python mfcc_gmm_train_test.py --test
     ```
   - The script will prompt you to enter your name and then record your voice for 5 seconds.
   - The script will then use the trained GMM model to recognize your voice and display the recognition score.

If you run the script without any arguments, it will display a message asking you to specify either `--train` or `--test`.

## How It Works

1. **Recording Voice Samples**:
   - The script uses the `pyaudio` library to record the user's voice for 5 seconds, 5 times.
   - The recorded voice samples are saved as WAV files in the `train_data` directory.

2. **Feature Extraction**:
   - The script uses the `librosa` library to extract MFCC features from the recorded voice samples.
   - MFCC features are a commonly used representation of audio signals in speech recognition and music processing.

3. **Training the GMM Model**:
   - The script uses the `scikit-learn` library to train a Gaussian Mixture Model (GMM) on the extracted MFCC features.
   - GMM is a probabilistic model that can be used to represent the distribution of the MFCC features for a given user's voice.

4. **Voice Recognition**:
   - During the test phase, the script records the user's voice and extracts the MFCC features.
   - The script then uses the trained GMM model to calculate the score of the recorded voice sample.
   - If the score is above a certain threshold, the script considers the voice to be recognized as the user.

## Future Improvements

- [ ] Fix recording problems:
  - [ ] Only create one audio file after recording five times.
  - [ ] Training process failure(TBC).
- [ ] Implement cross-validation to optimize the GMM model parameters.
- [ ] Explore other feature extraction techniques, such as Wavelet Transform or Deep Learning-based methods.
- [ ] Implement a more robust voice activity detection to improve the recording process.
- [ ] Add support for multiple users and user-specific models.