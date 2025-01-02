import csv
import os
import numpy as np
import librosa

# CSV dosyalarının adları ve yolları
emotion_csv_file = "csv_files/emotion_data.csv"
gender_csv_file = "csv_files/gender_data.csv"
age_csv_file = "csv_files/age_data.csv"

def create_emotion_dataset(directory_path):
    with open(emotion_csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5", "MFCC6", "MFCC7", "MFCC8", "MFCC9", "MFCC10",
                          "MFCC11", "MFCC12", "MFCC13", "MFCC14", "MFCC15", "MFCC16", "MFCC17", "MFCC18", "MFCC19", "MFCC20", 
                          "zero", "centroid", "rolloff"])
        for file in os.listdir(directory_path):
            if file.endswith(".flac"):
                file_path = os.path.join(directory_path, file)
                features = extract_features(file_path)
                if features is not None:
                    print(features)


def create_gender_dataset(directory_path):
    with open(gender_csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5", "MFCC6", "MFCC7", "MFCC8", "MFCC9", "MFCC10",
                          "MFCC11", "MFCC12", "MFCC13", "MFCC14", "MFCC15", "MFCC16", "MFCC17", "MFCC18", "MFCC19", "MFCC20",
                          "zero", "centroid", "rolloff"])        
        for file in os.listdir(directory_path):
            if file.endswith(".flac"):
                file_path = os.path.join(directory_path, file)
                features = extract_features(file_path)
                if features is not None:
                    writer.writerow(features)
                    print(features)

def create_age_dataset(directory_path):
    with open(age_csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5", "MFCC6", "MFCC7", "MFCC8", "MFCC9", "MFCC10",
                          "MFCC11", "MFCC12", "MFCC13", "MFCC14", "MFCC15", "MFCC16", "MFCC17", "MFCC18", "MFCC19", "MFCC20",
                            "zero","centroid","rolloff"])
        
        for file in os.listdir(directory_path):
            if file.endswith(".flac"):
                file_path = os.path.join(directory_path, file)
                features = extract_features(file_path)
                if features is not None:
                    print(features)


def extract_features(file_path, num_mfcc=20):
    try:
        # Ses dosyasını yükle
        signal, samplerate = librosa.load(file_path, sr=None)
        
        # MFCC özellikleri
        mfccs = librosa.feature.mfcc(y=signal, sr=samplerate, n_mfcc=num_mfcc)
        mfccs_avg = np.mean(mfccs.T, axis=0)
        
        # Ek özellikler hesapla
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(signal))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=samplerate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=samplerate))
        
        # Tüm özellikleri birleştir (20 MFCC + 3 ek özellik = 23 toplam)
        features = np.concatenate([
            mfccs_avg,
            [zero_crossing, spectral_centroid, spectral_rolloff]
        ])
        
        return features.tolist()
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Kullanım
create_emotion_dataset("./csv_files/")
create_gender_dataset("./csv_files/")
create_age_dataset("./csv_files/")
