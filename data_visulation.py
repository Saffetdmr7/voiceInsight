import pandas as pd
import matplotlib.pyplot as plt

# CSV dosyalarının yolları
age_csv = './csv_files/Yeni Klasör/age_only.csv'
gender_csv = './csv_files/Yeni Klasör/gender_only.csv'
emotion_csv = './csv_files/Yeni Klasör/emotion_only.csv'

# Verileri yükleme
age_data = pd.read_csv(age_csv)
gender_data = pd.read_csv(gender_csv)
emotion_data = pd.read_csv(emotion_csv)

# Sınıf dağılımı hesaplama
age_counts = age_data['age'].value_counts()
gender_counts = gender_data['gender'].value_counts()
emotion_counts = emotion_data['emotion'].value_counts()

# Dağılımı görselleştirme
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age dağılımı
age_counts.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Gender dağılımı
gender_counts.plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('Gender Distribution')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Count')

# Emotion dağılımı
emotion_counts.plot(kind='bar', ax=axes[2], color='lightgreen')
axes[2].set_title('Emotion Distribution')
axes[2].set_xlabel('Emotion')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.show()
