from emotion_extract_features import extract_features

# Duygu ve cinsiyet etiketleri
emotion_labels = {
    "0": "Neutral-Normal",
    "1": "Calm-Sakin",
    "2": "Happy-Mutlu",
    "3": "Sad-Üzgün",
    "4": "Angry-Kızgın",
    "5": "Fearful-Korkmuş",
    "6": "Disgust-Nefret",
    "7": "Surprised-Şaşkın"
}

gender_labels = {
    "1": "Female-Kadın",
    "0": "Male-Erkek"
}

age_labels = {
    "0": 'child(0-19)',
    "1": 'Young(20-35)',
    "2": 'Middle age(36-49)',
    "3": 'Old(49-90)'
}

import os
import joblib
import numpy as np
import pandas as pd

def predict(file_path, model_path):
    # Özellikleri çıkar
    features = extract_features(file_path)
    if features is None:
        return None, None, None
        
    # Özellik isimlerini güncelle - toplam 23 özellik
    feature_names = [f"MFCC{i+1}" for i in range(20)] + ['zero', 'centroid', 'rolloff']
    
    # Özellikleri düzenle
    features_flattened = np.array(features).reshape(1, -1)
    
    # Pandas DataFrame'e çevir ve özellik isimlerini ekle
    features_df = pd.DataFrame(features_flattened, columns=feature_names)
    
    try:
        # Modelleri yükle ve tahmin yap
        emotion_model = joblib.load(os.path.join(model_path, 'emotion_model_0.82.pkl'))
        gender_model = joblib.load(os.path.join(model_path, 'gender_model_0.96.pkl'))
        age_model = joblib.load(os.path.join(model_path, 'age_model_0.82.pkl'))
        
        # Tahminleri yap
        emotion_prediction = emotion_model.predict(features_df)[0]
        gender_prediction = gender_model.predict(features_df)[0]
        age_prediction = age_model.predict(features_df)[0]
        
        # Etiketleri al
        emotion_label = emotion_labels.get(str(emotion_prediction), "unknown")
        gender_label = gender_labels.get(str(gender_prediction), "unknown")
        age_label = age_labels.get(str(age_prediction), "unknown")
        
        return emotion_label, gender_label, age_label
        
    except Exception as e:
        print(f"Tahmin sırasında hata oluştu: {e}")
        return None, None, None

