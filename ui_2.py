import tkinter as tk
from tkinter import filedialog
import threading
import sounddevice as sd
import numpy as np
import os
from PIL import Image, ImageTk
import soundfile as sf
from main import predict

class ImageRoundedButton(tk.Button):
    def __init__(self, parent, image_path, text="", command=None, **kwargs):
        image = Image.open(image_path)
        image = image.resize((40, 40), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(image)
        super().__init__(parent, text=text, command=command, borderwidth=1, highlightthickness=0, compound='left',
                         image=self.img, padx=8, pady=5, anchor='w', **kwargs)
        self.image = self.img

class SesTahminiArayuzu:
    def __init__(self, ana_pencere):
        self.ana_pencere = ana_pencere
        self.ana_pencere.title("Ses Tahmini Arayüzü")
        self.ana_pencere.configure(bg="#f0f0f0")
        self.ana_pencere.geometry("500x560")
        button_conf = {'width': 180, 'height': 50, 'bg': "#f0f0f0"}
        # row0---------------------------------------
        self.dosya_secme_dugmesi = ImageRoundedButton(ana_pencere, "./icons/folder.png", text="Dosya ile",command=self.dosya_ile,  **button_conf)
        self.dosya_secme_dugmesi.grid(row=0, column=0, pady=(20, 10), padx=8, sticky='w')
        #row1---------------------------------------
        self.dosya_yolu_label = tk.Label(ana_pencere, text="", font=("Helvetica", 10, "bold"), bg="#f0f0f0")
        self.dosya_yolu_label.grid(row=1, column=0, pady=(20, 10), padx=8, sticky='w')
        # row2 col0---------------------------------------
        # Alt grid oluşturmak için frame kullanıyoruz
        self.kontrol_frame = tk.Frame(ana_pencere, bg="#f0f0f0")
        self.kontrol_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=8, sticky='nsew')

        self.kaydedilen_ses = []  # kaydedilen ses verilerini tutacak liste
        self.ses_kaydi_dugmesi = ImageRoundedButton(self.kontrol_frame, "./icons/play.png", text="Kaydı Başlat",command=self.ses_kaydet, **button_conf)
        self.ses_kaydi_dugmesi.grid(row=0, column=0, pady=10, padx=8, sticky='w')
        self.ses_kaydi_dugmesi.config(state=tk.NORMAL)

        self.kaydi_durdur_dugmesi = ImageRoundedButton(self.kontrol_frame, "./icons/stop-button.png", text="Kaydı Durdur",command=self.kaydi_durdur, **button_conf)
        self.kaydi_durdur_dugmesi.grid(row=1, column=0, pady=10, padx=8, sticky='w')
        self.kaydi_durdur_dugmesi.config(state=tk.DISABLED)

        self.kaydi_bitir_dugmesi = ImageRoundedButton(self.kontrol_frame, "./icons/stop-button.png", text="Kaydı Kaydet",command=self.kaydi_kaydet,  **button_conf)
        self.kaydi_bitir_dugmesi.grid(row=2, column=0, pady=10, padx=8, sticky='w')
        self.kaydi_bitir_dugmesi.config(state=tk.DISABLED)

        self.ses_kaydi_suresi_gostergesi = tk.Label(self.kontrol_frame, text="", font=("Helvetica", 10, "bold"), bg="#f0f0f0")
        self.ses_kaydi_suresi_gostergesi.grid(row=3, column=0, pady=(20, 10), padx=8, sticky='w')

        self.kayit_dosyasi_label = tk.Label(self.kontrol_frame, text="", font=("Helvetica", 10, "bold"), bg="#f0f0f0")
        self.kayit_dosyasi_label.grid(row=3, column=1, pady=(20, 10), padx=8, sticky='w')
        # row2 col1---------------------------------------
        self.emotion_label_text = tk.StringVar()
        self.emotion_label_text.set("Tahmin edilen duygu: ")
        self.emotion_label = tk.Label(self.kontrol_frame, textvariable=self.emotion_label_text, font=("Helvetica", 10, "bold"),bg="#f0f0f0", anchor='e')
        self.emotion_label.grid(row=0, column=1, pady=5, padx=8, sticky='e')

        self.gender_label_text = tk.StringVar()
        self.gender_label_text.set("Tahmin edilen cinsiyet: ")
        self.gender_label = tk.Label(self.kontrol_frame, textvariable=self.gender_label_text, font=("Helvetica", 10, "bold"),bg="#f0f0f0", anchor='e')
        self.gender_label.grid(row=1, column=1, pady=5, padx=8, sticky='e')

        self.age_label_text = tk.StringVar()
        self.age_label_text.set("Tahmin edilen yaş: ")
        self.age_label = tk.Label(self.kontrol_frame, textvariable=self.age_label_text, font=("Helvetica", 10, "bold"), bg="#f0f0f0",anchor='e')
        self.age_label.grid(row=2, column=1, pady=5, padx=8, sticky='e')
        # row3 col0---------------------------------------
        self.predict_frame = tk.Frame(ana_pencere, bg="#f0f0f0")
        self.predict_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=8, sticky='nsew')

        self.tahmin_et_dugmesi = ImageRoundedButton(self.predict_frame, "./icons/ai.png", text="Tahmin Et",command=self.tahmin_et, **button_conf)
        self.tahmin_et_dugmesi.grid(row=0, column=0, pady=10, padx=8, sticky='w')
        self.tahmin_et_dugmesi.config(state=tk.DISABLED)

        self.yeniden_tahmin_et_dugmesi = ImageRoundedButton(self.predict_frame, "./icons/ai.png", text="Yeniden Tahmin Et",                                                    command=self.yeniden_tahmin_et, **button_conf)
        self.yeniden_tahmin_et_dugmesi.grid(row=0, column=1, pady=10, padx=8, sticky='w')
        self.yeniden_tahmin_et_dugmesi.config(state=tk.DISABLED)

        self.kayit_aktif = False
        self.kayit_baslama_zamani = None
        self.kayit_suresi = 10
        self.ses_kaydi_durumu = "BASLA"
        self.tahmin_durumu = "ILK_TAHMIN"
        self.kayit_dosyasi = None
    def dosya_ile(self):
        dosya_adi = filedialog.askopenfilename(filetypes=[("FLAC files", "*.flac")])
        if dosya_adi:
            self.kayit_dosyasi = dosya_adi
            self.emotion_label_text.set("Tahmin edilen duygu: ")
            self.gender_label_text.set("Tahmin edilen cinsiyet: ")
            self.age_label_text.set("Tahmin edilen yaş: ")
            self.tahmin_et_dugmesi.config(state=tk.NORMAL)
            self.dosya_yolu_label.config(text="Seçilen dosya: " + dosya_adi)
            self.ses_kaydi_dugmesi.config(state=tk.DISABLED)
    def ses_kaydet(self):
        if not self.kayit_aktif:
            self.kayit_aktif = True
            self.ses_kaydi_dugmesi.config(text="Ses Kaydına Devam Et", state=tk.DISABLED)
            self.kaydi_durdur_dugmesi.config(state=tk.NORMAL)
            self.kaydi_bitir_dugmesi.config(state=tk.DISABLED)
            self.kaydi_baslama_zamani = 44100
            self.ses_kaydi_suresi_gostergesi.config(text="Ses kaydı başladı...", fg="#009688")
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
        self.dosya_secme_dugmesi.config(state=tk.DISABLED)
    def kaydi_durdur(self):
        self.kayit_aktif = False
        self.ses_kaydi_dugmesi.config(text="Ses Kaydına Başla", state=tk.NORMAL)
        self.kaydi_durdur_dugmesi.config(state=tk.DISABLED)
        self.kaydi_bitir_dugmesi.config(state=tk.NORMAL)
        self.ses_kaydi_suresi_gostergesi.config(text="Ses kaydı durduruldu.", fg="#f44336")
    def kaydi_kaydet(self):
        self.kayit_aktif = False
        if self.kaydedilen_ses:
            dosya_adi = "kaydedilen_ses.flac"
            sf.write(dosya_adi, np.concatenate(self.kaydedilen_ses), 44100, format='flac')
            self.kayit_dosyasi = dosya_adi
            self.ses_kaydi_suresi_gostergesi.config(text="Ses kaydı başarıyla kaydedildi.", fg="#4caf50")
            self.kaydi_bitir_dugmesi.config(state=tk.DISABLED)
            self.tahmin_et_dugmesi.config(state=tk.NORMAL)
            self.kayit_dosyasi_label.config(text="Kaydedilen dosya: " + dosya_adi)
        else:
            self.ses_kaydi_suresi_gostergesi.config(text="Kaydedilen ses bulunamadı.", fg="#f44336")
    def tahmin_et(self):
        if self.kayit_dosyasi and os.path.exists(self.kayit_dosyasi):
            emotion_label, gender_label, age_label = predict(self.kayit_dosyasi, "./models")
            self.emotion_label_text.set("Tahmin edilen duygu: " + emotion_label)
            self.gender_label_text.set("Tahmin edilen cinsiyet: " + gender_label)
            self.age_label_text.set("Tahmin edilen yaş: " + age_label)
            self.yeniden_tahmin_et_dugmesi.config(state=tk.NORMAL)

        else:
            print("Kayıt dosyası bulunamadı.")
    def yeniden_tahmin_et(self):
        self.emotion_label_text.set("Tahmin edilen duygu: ")
        self.gender_label_text.set("Tahmin edilen cinsiyet: ")
        self.age_label_text.set("Tahmin edilen yaş: ")
        self.kayit_dosyasi = None
        self.tahmin_et_dugmesi.config(state=tk.DISABLED)
        self.kayit_dosyasi_label.config(text="")
        self.dosya_yolu_label.config(text="")
        self.ses_kaydi_dugmesi.config(state=tk.NORMAL)
        self.dosya_secme_dugmesi.config(state=tk.NORMAL)
        self.ses_kaydi_suresi_gostergesi.config(text="")


ana_pencere = tk.Tk()
ses_tahmini_arayuzu = SesTahminiArayuzu(ana_pencere)
ana_pencere.mainloop()
