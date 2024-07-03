import os
import pyaudio
import wave

 # Use pyaudio to record the voice
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000


def record(path, RECORD_SECONDS=5):

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
        )
    
    frames = [] # Initialize an empty list to store the audio data

    # Record the audio in chunks
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Close the stream and terminate the pyaudio object
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio data to a .wav file
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return path

