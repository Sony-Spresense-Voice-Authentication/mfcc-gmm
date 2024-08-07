# Voice Recognition using MFCC and GMM

This Python script allows you to train and test a voice recognition model using Mel-Frequency Cepstral Coefficients (MFCC) and Gaussian Mixture Models (GMM).

# This project is now under construction, and the model is not working properly.

## Requirements

The script requires the following Python libraries:

- `librosa`: A Python package for audio and music analysis
- `numpy`: A fundamental library for scientific computing in Python
- `scikit-learn`: A machine learning library for Python
- `pyaudio`: A Python binding for the PortAudio library, used for audio I/O

You can install these dependencies by running the following command when you use Windows:

```
pip install -r src/requirements.txt
pip install pyaudio
```

Specifically for MacOS, you should install `portaudio` before installing `PyAudio`:

```zsh
brew install portaudio
pip install pyaudio
```

For more systems just refer to https://pypi.org/project/PyAudio/ .

## Usage


### 1. **Training the Model**:

   - Run the script:
     ```python
     python src/main.py 
     ```
   - The script will prompt you to enter your name, and then record your voice for 5 seconds, 5 times by default.
   - The recorded voice samples will be saved as WAV files in the `audio` directory.
   - You can also specify the number of recordings and the duration of each recording using the `--num_recordings`,  `--phase` and `--duration` arguments:
     ```python
     python src/main.py --num_recordings 10 --duration 10 --phase "Test phase for several seconds."
     ```

### 2. **Testing the Model**:
   - Run the script with the `-a` or `--auth` argument:
     ```python
     python src/main.py -a
     ```
   - The script will prompt you to enter your name and then record your voice for 5 seconds.
   - The script will then use the trained GMM model to recognize your voice and display the recognition score.

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

- [x] Add support for recording multiple voice samples for training.
- [ ] Fix the recording process to avoid creating multiple audio files.
- [ ] Model tuning problems:
  - [ ] The model return every sample as authenticated.
  - [ ] Threshold tuning.
  - [ ] Cross-validation to optimize the GMM model parameters.
- [x] Fix recording problems:
  - [x] Only create one audio file after recording five times.
  - [x] Training process failure(TBC).
- [ ] Explore other feature extraction techniques, such as Wavelet Transform or Deep Learning-based methods.
- [ ] Implement a more robust voice activity detection to improve the recording process.
- [x] Add support for multiple users and user-specific models.(Not tested.)