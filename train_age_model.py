import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import numpy as np


# Cross-validation stratejisini uygulandı
# FitFailedWarning ve diğer olası uyarıları bastır
warnings.filterwarnings('ignore', category=UserWarning)

# Veriyi yükleyin
data = pd.read_csv('./csv_files/Yeni klasör/age_only.csv')

# Hedef değişken
target = 'age'

# Özellik ve hedef değişkenleri ayırın
X = data.drop(columns=target)
y = data[target]

# Veri setini %70 eğitim ve %30 test+doğrulama olarak bölün
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Test+doğrulama setini %50 doğrulama ve %50 test olarak bölün
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Veri oranlarını kontrol et
print(f"Eğitim seti boyutu: {len(X_train)}")
print(f"Doğrulama seti boyutu: {len(X_val)}")
print(f"Test seti boyutu: {len(X_test)}")

# Başlangıç hiperparametre aralıkları
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

best_accuracy = 0
best_model = None
best_params = None
results = []

# Hiperparametre aralığını daraltarak iteratif olarak Grid Search
for i in range(5):  # 5 iterasyonla sınırlı
    print(f"Iteration {i+1}")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise')
    grid_search.fit(X_train, y_train)
    
    current_model = grid_search.best_estimator_
    current_params = grid_search.best_params_
    y_pred = current_model.predict(X_val)
    current_accuracy = accuracy_score(y_val, y_pred)
    
    results.append({
        'iteration': i + 1,
        'params': current_params,
        'accuracy': current_accuracy
    })
    
    print(f"Best params for iteration {i+1}: {current_params}")
    print(f"Validation Accuracy for iteration {i+1}: {current_accuracy:.2f}")
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model = current_model
        best_params = current_params
        
        # Başarılı parametrelerin etrafında daha detaylı arama
        param_grid = {
            'n_estimators': [
                max(50, current_params['n_estimators'] - 100),
                current_params['n_estimators'],
                current_params['n_estimators'] + 100
            ],
            'max_depth': [
                max(5, current_params['max_depth'] - 5),
                current_params['max_depth'],
                current_params['max_depth'] + 5
            ],
            'min_samples_split': [
                max(2, current_params['min_samples_split'] - 2),
                current_params['min_samples_split'],
                current_params['min_samples_split'] + 2
            ],
            'min_samples_leaf': [
                max(1, current_params['min_samples_leaf'] - 1),
                current_params['min_samples_leaf'],
                current_params['min_samples_leaf'] + 1
            ]
        }
    else:
        # Performans düşüşünde farklı bir bölgede arama yap
        param_grid = {
            'n_estimators': [150, 250, 350],  # Farklı değerler dene
            'max_depth': [15, 25, 35],        # Farklı değerler dene
            'min_samples_split': [3, 6, 9],   # Farklı değerler dene
            'min_samples_leaf': [1, 3, 5]     # Farklı değerler dene
        }

# Modeli ve doğruluğunu .pkl dosyasına kaydetme
model_filename = f'age_model_{best_accuracy:.2f}.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Best model saved as {model_filename} with accuracy: {best_accuracy:.2f}")
print(f"Best parameters: {best_params}")
# Sonuçları bir DataFrame'e dönüştür
results_df = pd.DataFrame(results)

# Test seti üzerinde tahmin yap
y_test_pred = best_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Confusion Matrix görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Sınıflandırma Raporu
class_report = classification_report(y_test, y_test_pred, target_names=[str(label) for label in np.unique(y)])
print("Sınıflandırma Raporu:")
print(class_report)

# İterasyon başına doğruluk değerlerini görselleştirme
# Sonuçları bir DataFrame'e dönüştür (eğer sonuçlar eksikse bu adım hatayı düzeltecektir)
results_df = pd.DataFrame(results)

if 'iteration' in results_df.columns and 'accuracy' in results_df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['iteration'], results_df['accuracy'], marker='o', linestyle='-', color='b', label="Validation Accuracy")
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Iteration')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("Hata: 'iteration' veya 'accuracy' sütunları sonuç DataFrame'inde mevcut değil!")
    print("Lütfen sonuçların doğru bir şekilde kaydedildiğinden emin olun.")

# Her iterasyondaki en iyi parametreleri ve doğruluk değerlerini yazdırma
for result in results:
    print(f"Iteration {result['iteration']}:")
    print(f"Params: {result['params']}")
    print(f"Validation Accuracy: {result['accuracy']:.2f}")
    print("")

print(f"Best parameters: {best_params}, Best validation accuracy: {best_accuracy:.2f}")
