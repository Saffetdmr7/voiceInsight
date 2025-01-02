import customtkinter as ctk
from tkinter import filedialog
import threading
import sounddevice as sd
import numpy as np
import os
from PIL import Image
import soundfile as sf
from main import predict

# Tema ayarları
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class ModernButton(ctk.CTkButton):
    def __init__(self, parent, image_path, text="", command=None, **kwargs):
        image = Image.open(image_path)
        image = image.resize((24, 24), Image.LANCZOS)
        
        super().__init__(
            parent,
            text=text,
            command=command,
            image=ctk.CTkImage(light_image=image, dark_image=image),
            compound="left",
            font=("Segoe UI", 12),
            corner_radius=8,
            border_spacing=10,
            fg_color="#2c3e50",
            hover_color="#34495e",
            **kwargs
        )

class SesTahminiArayuzu:
    def __init__(self, ana_pencere):
        self.ana_pencere = ana_pencere
        self.ana_pencere.title("Ses Analiz Sistemi")
        self.ana_pencere.geometry("800x700")
        
        # Ana container
        self.main_container = ctk.CTkFrame(ana_pencere, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        ctk.CTkLabel(
            self.main_container,
            text="Ses Analiz Sistemi",
            font=("Segoe UI", 24, "bold"),
            text_color="#2c3e50"
        ).pack(pady=(0, 30))

        # Sol panel - Kontroller
        self.sol_panel = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.sol_panel.pack(side="left", fill="y", padx=(0, 10))

        # Dosya seçme bölümü
        self.dosya_frame = ctk.CTkFrame(self.sol_panel, fg_color="transparent")
        self.dosya_frame.pack(fill="x", pady=10, padx=10)
        
        self.dosya_secme_dugmesi = ModernButton(
            self.dosya_frame,
            "./icons/folder.png",
            text="Dosya Seç",
            command=self.dosya_ile,
            width=200
        )
        self.dosya_secme_dugmesi.pack(pady=5)

        # Kayıt kontrol butonları
        self.kayit_kontrol_frame = ctk.CTkFrame(self.sol_panel, fg_color="transparent")
        self.kayit_kontrol_frame.pack(fill="x", pady=10, padx=10)

        self.ses_kaydi_dugmesi = ModernButton(
            self.kayit_kontrol_frame,
            "./icons/play.png",
            text="Kaydı Başlat",
            command=self.ses_kaydet,
            width=200
        )
        self.ses_kaydi_dugmesi.pack(pady=5)

        self.kaydi_durdur_dugmesi = ModernButton(
            self.kayit_kontrol_frame,
            "./icons/stop-button.png",
            text="Kaydı Durdur",
            command=self.kaydi_durdur,
            width=200,
            state="disabled"
        )
        self.kaydi_durdur_dugmesi.pack(pady=5)

        self.kaydi_bitir_dugmesi = ModernButton(
            self.kayit_kontrol_frame,
            "./icons/stop-button.png",
            text="Kaydı Kaydet",
            command=self.kaydi_kaydet,
            width=200,
            state="disabled"
        )
        self.kaydi_bitir_dugmesi.pack(pady=5)

        # Tahmin butonları
        self.tahmin_frame = ctk.CTkFrame(self.sol_panel, fg_color="transparent")
        self.tahmin_frame.pack(fill="x", pady=10, padx=10)

        self.tahmin_et_dugmesi = ModernButton(
            self.tahmin_frame,
            "./icons/ai.png",
            text="Tahmin Et",
            command=self.tahmin_et,
            width=200,
            state="disabled"
        )
        self.tahmin_et_dugmesi.pack(pady=5)

        self.yeniden_tahmin_et_dugmesi = ModernButton(
            self.tahmin_frame,
            "./icons/ai.png",
            text="Yeniden Tahmin Et",
            command=self.yeniden_tahmin_et,
            width=200,
            state="disabled"
        )
        self.yeniden_tahmin_et_dugmesi.pack(pady=5)

        # Dosya yolu ve kayıt dosyası etiketleri
        self.dosya_yolu_label = ctk.CTkLabel(
            self.sol_panel,
            text="",
            font=("Segoe UI", 10),
            text_color="#7f8c8d"
        )
        self.dosya_yolu_label.pack(pady=5)

        self.kayit_dosyasi_label = ctk.CTkLabel(
            self.sol_panel,
            text="",
            font=("Segoe UI", 10),
            text_color="#7f8c8d"
        )
        self.kayit_dosyasi_label.pack(pady=5)

        # Sağ panel - Sonuçlar
        self.sag_panel = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.sag_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Sonuç kartları
        self.sonuc_baslik = ctk.CTkLabel(
            self.sag_panel,
            text="Tahmin Sonuçları",
            font=("Segoe UI", 18, "bold"),
            text_color="#2c3e50"
        )
        self.sonuc_baslik.pack(pady=15)

        # Sonuç değişkenleri
        self.emotion_label_text = ctk.StringVar(value="Duygu: Bekleniyor...")
        self.gender_label_text = ctk.StringVar(value="Cinsiyet: Bekleniyor...")
        self.age_label_text = ctk.StringVar(value="Yaş: Bekleniyor...")

        # Sonuç kartları oluştur
        self.create_result_cards()

        # Durum göstergesi
        self.durum_frame = ctk.CTkFrame(self.main_container, height=40, corner_radius=8)
        self.durum_frame.pack(side="bottom", fill="x", pady=(20, 0))
        
        self.ses_kaydi_suresi_gostergesi = ctk.CTkLabel(
            self.durum_frame,
            text="",
            font=("Segoe UI", 12)
        )
        self.ses_kaydi_suresi_gostergesi.pack(pady=10)

        # Değişkenler
        self.kayit_aktif = False
        self.kayit_dosyasi = None
        self.kaydedilen_ses = []

    def create_result_cards(self):
        for label_text in [self.emotion_label_text, self.gender_label_text, self.age_label_text]:
            card = ctk.CTkFrame(self.sag_panel, corner_radius=10)
            card.pack(fill="x", padx=15, pady=5)
            
            ctk.CTkLabel(
                card,
                textvariable=label_text,
                font=("Segoe UI", 14),
                pady=15
            ).pack()

    def dosya_ile(self):
        dosya_adi = filedialog.askopenfilename(filetypes=[("FLAC files", "*.flac")])
        if dosya_adi:
            self.kayit_dosyasi = dosya_adi
            self.emotion_label_text.set("Tahmin edilen duygu: ")
            self.gender_label_text.set("Tahmin edilen cinsiyet: ")
            self.age_label_text.set("Tahmin edilen yaş: ")
            self.ses_kaydi_dugmesi.configure(state="disabled")
            self.dosya_yolu_label.configure(text="Seçilen dosya: " + dosya_adi)
    def ses_kaydet(self):
        if not self.kayit_aktif:
            self.kayit_aktif = True
            self.ses_kaydi_dugmesi.configure(text="Ses Kaydına Devam Et", state="disabled")
            self.kaydi_durdur_dugmesi.configure(state="normal")
            self.kaydi_bitir_dugmesi.configure(state="disabled")
            self.kaydi_baslama_zamani = 44100
            self.ses_kaydi_suresi_gostergesi.configure(text="Ses kaydı başladı...", text_color="#009688")
            self.kayit_thread = threading.Thread(target=self.ses_kaydet_basla)
            self.kayit_thread.start()
        else:
            self.ses_kaydet_basla()
    def ses_kaydet_basla(self):
        self.kaydedilen_ses = []
        def kaydet_callback(indata, frames, time, status):
            if status:
                print(status)
            self.kaydedilen_ses.append(indata.copy())
        with sd.InputStream(callback=kaydet_callback, samplerate=44100, channels=2):
            while self.kayit_aktif:
                sd.sleep(1000)
        self.dosya_secme_dugmesi.configure(state="disabled")
    def kaydi_durdur(self):
        self.kayit_aktif = False
        self.ses_kaydi_dugmesi.configure(text="Ses Kaydına Başla", state="normal")
        self.kaydi_durdur_dugmesi.configure(state="disabled")
        self.kaydi_bitir_dugmesi.configure(state="normal")
        self.ses_kaydi_suresi_gostergesi.configure(text="Ses kaydı durduruldu.", text_color="#f44336")
    def kaydi_kaydet(self):
        self.kayit_aktif = False
        if self.kaydedilen_ses:
            dosya_adi = "kaydedilen_ses.flac"
            sf.write(dosya_adi, np.concatenate(self.kaydedilen_ses), 44100, format='flac')
            self.kayit_dosyasi = dosya_adi
            self.ses_kaydi_suresi_gostergesi.configure(text="Ses kaydı başarıyla kaydedildi.", text_color="#4caf50")
            self.kaydi_bitir_dugmesi.configure(state="disabled")
            self.tahmin_et_dugmesi.configure(state="normal")
            self.kayit_dosyasi_label.configure(text="Kaydedilen dosya: " + dosya_adi)
        else:
            self.ses_kaydi_suresi_gostergesi.configure(text="Kaydedilen ses bulunamadı.", text_color="#f44336")
    def tahmin_et(self):
        if self.kayit_dosyasi and os.path.exists(self.kayit_dosyasi):
            emotion_label, gender_label, age_label = predict(self.kayit_dosyasi, "./models")
            self.emotion_label_text.set("Tahmin edilen duygu: " + emotion_label)
            self.gender_label_text.set("Tahmin edilen cinsiyet: " + gender_label)
            self.age_label_text.set("Tahmin edilen yaş: " + age_label)
            self.yeniden_tahmin_et_dugmesi.configure(state="normal")
        else:
            print("Kayıt dosyası bulunamadı.")
    def yeniden_tahmin_et(self):
        self.emotion_label_text.set("Tahmin edilen duygu: ")
        self.gender_label_text.set("Tahmin edilen cinsiyet: ")
        self.age_label_text.set("Tahmin edilen yaş: ")
        self.kayit_dosyasi = None
        self.tahmin_et_dugmesi.configure(state="disabled")
        self.kayit_dosyasi_label.configure(text="")
        self.dosya_yolu_label.configure(text="")
        self.ses_kaydi_dugmesi.configure(state="normal")
        self.dosya_secme_dugmesi.configure(state="normal")
        self.ses_kaydi_suresi_gostergesi.configure(text="")


ana_pencere = ctk.CTk()
ses_tahmini_arayuzu = SesTahminiArayuzu(ana_pencere)
ana_pencere.mainloop()
