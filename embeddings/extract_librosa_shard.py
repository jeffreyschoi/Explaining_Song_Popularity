import sys
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

shard_id = int(sys.argv[1])
netID = "jsc9862"

input_path = f"/scratch/{netID}/Explaining_Song_Popularity/data/shards/songs_shard_{shard_id}.csv"
output_path = f"/scratch/{netID}/Explaining_Song_Popularity/data/librosa_shard_{shard_id}.pkl"

df = pd.read_csv(input_path)

def extract_librosa_features(path):
    try:
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_stft_mean = chroma_stft.mean(axis=1)
        chroma_cqt_mean = chroma_cqt.mean(axis=1)
        chroma_cens_mean = chroma_cens.mean(axis=1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = spectral_contrast.mean(axis=1)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = tonnetz.mean(axis=1)


        feat = {
            "duration": duration,
            "rms_energy": rms_energy,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "zero_crossing_rate": zero_crossing_rate,
            "tempo": tempo,
        }

        # Chroma (12 dim each)
        for i in range(12):
            feat[f"chroma_stft_{i+1}"] = chroma_stft_mean[i]
            feat[f"chroma_cqt_{i+1}"] = chroma_cqt_mean[i]
            feat[f"chroma_cens_{i+1}"] = chroma_cens_mean[i]

        # MFCC mean + std (13 each)
        for i in range(13):
            feat[f"mfcc_mean_{i+1}"] = mfcc_mean[i]
            feat[f"mfcc_std_{i+1}"] = mfcc_std[i]

        # Spectral contrast (librosa usually returns 7 bands)
        for i in range(len(spectral_contrast_mean)):
            feat[f"spectral_contrast_{i+1}"] = spectral_contrast_mean[i]

        # Tonnetz (6 dim)
        for i in range(6):
            feat[f"tonnetz_{i+1}"] = tonnetz_mean[i]

        return feat

    except Exception as e:
        return {"error": str(e)}

feature_dicts = []
for p in tqdm(df["file_path"], desc=f"Shard {shard_id}"):
    feature_dicts.append(extract_librosa_features(p))

features_df = pd.DataFrame(feature_dicts)

merged = pd.concat([df.reset_index(drop=True), features_df], axis=1)

merged.to_pickle(output_path)
print("Saved:", output_path)
