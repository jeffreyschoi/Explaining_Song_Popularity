#!/usr/bin/env python3
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import os
import sys

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_librosa_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # RMS
        rms = librosa.feature.rms(y=y).mean()

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()

        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

        # Chroma
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr).mean()
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr).mean()

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfcc.mean(axis=1)
        mfcc_stds = mfcc.std(axis=1)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = contrast.mean(axis=1)

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = tonnetz.mean(axis=1)

        features = {
            "duration": duration,
            "rms_energy": rms,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "zero_crossing_rate": zcr,
            "tempo": tempo,
            "chroma_stft_mean": chroma_stft,
            "chroma_cqt_mean": chroma_cqt,
            "chroma_cens_mean": chroma_cens,
        }

        # Add MFCCs
        for i in range(13):
            features[f"mfcc_mean_{i+1}"] = mfcc_means[i]
            features[f"mfcc_std_{i+1}"] = mfcc_stds[i]

        # Add contrast
        for i, val in enumerate(contrast_mean):
            features[f"spectral_contrast_{i+1}"] = val

        # Add tonnetz
        for i, val in enumerate(tonnetz_mean):
            features[f"tonnetz_{i+1}"] = val

        return features

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Main processing script
# -----------------------------
if __name__ == "__main__":
    netID = os.getenv("USER", "jsc9862")
    csv_path = f"/scratch/{netID}/Explaining_Song_Popularity/final_cleaned_songs.csv"

    print("Loading CSV:", csv_path)
    songs_df = pd.read_csv(csv_path)

    feature_dicts = []
    file_paths = songs_df["file_path"].tolist()

    print("Extracting librosa features...")
    for path in tqdm(file_paths):
        feature_dicts.append(extract_librosa_features(path))

    features_df = pd.DataFrame(feature_dicts)
    songs_with_audio_feats = pd.concat([
        songs_df.reset_index(drop=True),
        features_df.reset_index(drop=True)
    ], axis=1)

    out_path = f"/scratch/{netID}/Explaining_Song_Popularity/songs_with_librosa_features.pkl"
    print("Saving to:", out_path)
    songs_with_audio_feats.to_pickle(out_path)

    print("Done.")
