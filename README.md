# 🎙️ Ses Analiz Sistemi

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-orange.svg)
![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-up%20to%2096%25-success.svg)

Ses kayıtlarından yapay zeka kullanarak duygu, yaş ve cinsiyet analizi yapan gelişmiş bir sistem.

[Özellikler](#-özellikler) •
[Kurulum](#-kurulum) •
[Kullanım](#-kullanım) •
[Teknolojiler](#-teknolojiler) •
[Modeller](#-modeller)

</div>

---

## 📋 İçindekiler
1. [Proje Hakkında](#-proje-hakkında)
2. [Özellikler](#-özellikler)
3. [Kurulum](#-kurulum)
4. [Kullanım](#-kullanım)
5. [Teknolojiler](#-teknolojiler)
6. [Modeller](#-modeller)
7. [Katkıda Bulunma](#-katkıda-bulunma)
8. [İletişim](#-iletişim)

## 🎯 Proje Hakkında

Bu proje, ses kayıtlarını analiz ederek konuşmacının duygusal durumunu, yaşını ve cinsiyetini tahmin eden yapay zeka tabanlı bir sistemdir. Kullanıcı dostu arayüzü ile hem canlı ses kaydı alabilir hem de var olan ses dosyalarını analiz edebilirsiniz.

## 🌟 Özellikler

### 🎭 Duygu Analizi
- **8 Farklı Duygu Durumu**
  - 😐 Nötr (Neutral)
  - 😌 Sakin (Calm)
  - 😊 Mutlu (Happy)
  - 😢 Üzgün (Sad)
  - 😠 Kızgın (Angry)
  - 😨 Korkmuş (Fearful)
  - 😤 Nefret (Disgust)
  - 😲 Şaşkın (Surprised)
- **Doğruluk Oranı**: 82%

### 👥 Cinsiyet Tahmini
- **İki Kategori**
  - 👩 Kadın (Female)
  - 👨 Erkek (Male)
- **Doğruluk Oranı**: 96%

### 👶👴 Yaş Tahmini
- **4 Yaş Grubu**
  - 👶 Çocuk (0-19)
  - 👱 Genç (20-35)
  - 👨 Orta Yaş (36-49)
  - 👴 Yaşlı (49-90)
- **Doğruluk Oranı**: 82%

## 🚀 Kurulum

1. **Depoyu Klonlayın**

```
git clone https://github.com/Saffetdmr7/voiceInsight.git
cd voiceInsight
```

2. **Sanal Ortam Oluşturun ve Aktifleştirin**
```bash
python -m venv venv
# Windows için
venv\Scripts\activate
# Linux/Mac için
source venv/bin/activate
```

3. **Gereksinimleri Yükleyin**
```bash
pip install -r requirements.txt
```

## 💻 Kullanım

### 1. Canlı Ses Kaydı ile Analiz
1. Programı başlatın: `python ui_2.py`
2. "Kaydı Başlat" butonuna tıklayın
3. Konuşmaya başlayın
4. "Kaydı Durdur" ile kaydı sonlandırın
5. "Tahmin Et" ile analizi başlatın

### 2. Dosyadan Analiz
1. "Dosya Seç" butonuna tıklayın
2. .flac formatındaki ses dosyasını seçin
3. "Tahmin Et" ile sonuçları görüntüleyin

## 🛠 Teknolojiler

- **Temel Teknolojiler**
  - Python 3.11
  - CustomTkinter (Modern GUI)
  - NumPy (Veri İşleme)
  - Pandas (Veri Manipülasyonu)

- **Ses İşleme**
  - SoundDevice (Ses Kaydı)
  - SoundFile (Ses Dosyası İşlemleri)
  - Librosa (Ses Özellik Çıkarımı)

- **Yapay Zeka**
  - Scikit-learn (Makine Öğrenmesi)
  - Joblib (Model Yükleme)

## 📊 Modeller

Sistemde kullanılan modeller Random Forest algoritması ile eğitilmiştir:

| Model | Doğruluk | Dosya Adı |
|-------|----------|-----------|
| Duygu | 82% | emotion_model_0.82.pkl |
| Cinsiyet | 96% | gender_model_0.96.pkl |
| Yaş | 82% | age_model_0.82.pkl |

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: XYZ'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

---

<div align="center">

### ⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

</div>