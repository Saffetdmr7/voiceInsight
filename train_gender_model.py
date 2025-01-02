import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle


# Uyarıları bastır
warnings.filterwarnings('ignore', category=UserWarning)

# Veriyi yükleyin
data = pd.read_csv('./csv_files/Yeni klasör/gender_only.csv')


# Hedef değişken
target = 'gender'

# Özellik ve hedef değişkenleri ayırın
X = data.drop(columns=target)
y = data[target]

# Hedef değişken sınıf dağılımını kontrol edin
print("Hedef değişken dağılımı:")
print(y.value_counts(normalize=True))

# Veri setini eğitim (%70), doğrulama (%15) ve test (%15) olarak ayırın
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Veri oranlarını kontrol et
print(f"Eğitim seti boyutu: {len(X_train)}")
print(f"Doğrulama seti boyutu: {len(X_val)}")
print(f"Test seti boyutu: {len(X_test)}")

# Eğitim ve doğrulama setlerinin bağımsız olduğunu kontrol edin
assert set(X_train.index).isdisjoint(X_val.index), "Eğitim ve doğrulama setleri kesişiyor!"
assert set(X_train.index).isdisjoint(X_test.index), "Eğitim ve test setleri kesişiyor!"
assert set(X_val.index).isdisjoint(X_test.index), "Doğrulama ve test setleri kesişiyor!"

# Başlangıç hiperparametre aralıkları
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

best_accuracy = 0
best_model = None
best_params = None
results = []

# Hedef değişken sınıf dengesizliği için class weight hesaplama
class_weights = compute_class_weight(class_weight='balanced', classes=y.unique(), y=y)
class_weights_dict = dict(zip(y.unique(), class_weights))

# Hiperparametre optimizasyonu
for i in range(5):  # 5 iterasyonla sınırlı
    print(f"\nIterasyon {i+1}")
    rf = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)  # Dengesiz sınıflar için ağırlık eklendi
    
    # Grid Search ile en iyi hiperparametreleri bulma
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    
    # En iyi modeli ve parametreleri al
    current_model = grid_search.best_estimator_
    current_params = grid_search.best_params_
    
    # Validation seti üzerinde değerlendir
    y_val_pred = current_model.predict(X_val)
    current_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Sonuçları kaydet
    results.append({
        'iteration': i + 1,
        'params': current_params,
        'accuracy': current_accuracy,
        'cv_score': grid_search.best_score_
    })
    
    print(f"En iyi parametreler: {current_params}")
    print(f"Cross-validation skoru: {grid_search.best_score_:.4f}")
    print(f"Validation doğruluğu: {current_accuracy:.4f}")
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model = current_model
        best_params = current_params
        

        # Parametre aralığını daralt
        param_grid = {
            'n_estimators': [
                max(50, current_params['n_estimators'] - 50),
                current_params['n_estimators'],
                current_params['n_estimators'] + 50
            ],
            'max_depth': [
                max(5, current_params['max_depth'] - 3),
                current_params['max_depth'],
                current_params['max_depth'] + 3
            ],
            'min_samples_split': [
                max(2, current_params['min_samples_split'] - 1),
                current_params['min_samples_split'],
                min(20, current_params['min_samples_split'] + 1)
            ],
            'min_samples_leaf': [
                max(1, current_params['min_samples_leaf'] - 1),
                current_params['min_samples_leaf'],
                min(10, current_params['min_samples_leaf'] + 1)
            ]
        }
    else:
        # Performans düşerse farklı bir bölgede ara
        param_grid = {
            'n_estimators': [150, 250, 350],
            'max_depth': [15, 25, 35],
            'min_samples_split': [3, 6, 9],
            'min_samples_leaf': [1, 3, 5]
        }

# Final test performansını değerlendir
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nSonuçlar:")
print(f"En iyi parametreler: {best_params}")
print(f"Validation doğruluğu: {best_accuracy:.4f}")
print(f"Test doğruluğu: {test_accuracy:.4f}")

# Confusion matrix ve sınıflandırma raporu
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))

# Modeli kaydet
model_filename = f'gender_model_{test_accuracy:.2f}.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Model başarıyla kaydedildi: {model_filename}")

# İterasyon başına doğruluk grafiği
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(results_df['iteration'], results_df['accuracy'], marker='o', linestyle='-', color='b', label='Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracy per Iteration')
plt.legend()
plt.grid(True)
plt.show()

# Her iterasyondaki en iyi parametreleri yazdırma
for result in results:
    print(f"Iteration {result['iteration']}:")
    print(f"Params: {result['params']}")
    print(f"Accuracy: {result['accuracy']:.2f}\n")


# Eğitim ve test setlerinin kesişim kontrolü
print(set(X_val.index).intersection(set(X_test.index)))
