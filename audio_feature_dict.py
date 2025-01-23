import essentia.standard as es
import librosa
import pickle
import os
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

feature_dict = {}
songs_folder = "3 ugers/Taylor_Swift_songs"

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Read the song names from the file
with open("3 ugers/Song_list.txt") as file:
    song_list = [line.rstrip() for line in file]

# Loop through all songs from song_list
for file_name in song_list:
    song = file_name + ".mp3"
    song_path = os.path.join(songs_folder, song)
    
    # Load the audio as a waveform 'y' and store the sampling rate in 'sr'
    y, sr = librosa.load(song_path, offset=10.0)

    # Mean and variance of ...
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Mean and variance of ...
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_flatness_var = np.var(librosa.feature.spectral_flatness(y=y))

    # Mean and variance of ...
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Mean and variance of ...
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    spectral_contrast_var = np.var(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # Estimate BPM
    BPM, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Load audio and initialize the KeyExtractor
    audio = es.MonoLoader(filename=song_path)()
    key_extractor = es.KeyExtractor()

    # Returns (key, scale, strength)
    detected_key, scale, strength = key_extractor(audio)

    # Change scale from string to integer
    if scale == "major":
        scale = 1
    else:
        scale = -1 

    lyric_path = os.path.join("3 ugers/Lyrics", file_name + ".txt")
     # Read the lyrics from the file
    with open(lyric_path, "r") as file:
        text = file.read()
    
    # Perform sentiment analysis
    score = sia.polarity_scores(text)
    
    # Save results in dictionary
    feature_dict[file_name] = [BPM[0], scale, score["compound"],
                               spectral_centroid, spectral_flatness, spectral_bandwidth, spectral_contrast,
                               spectral_centroid_var, spectral_flatness_var, spectral_bandwidth_var, spectral_contrast_var]



# Save the dictionary
with open("feature_dict.pkl", "wb") as f:
    pickle.dump(feature_dict, f)

print("done!")