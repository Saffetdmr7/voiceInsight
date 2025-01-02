import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

# FitFailedWarning ve diğer olası uyarıları bastır
warnings.filterwarnings('ignore', category=UserWarning)

def grid_search_optimization(data_path, target, max_iterations=10):
    # Veriyi yükleyin
    data = pd.read_csv(data_path)

    # Özellik ve hedef değişkenleri ayırın
    X = data.drop(columns=target)
    y = data[target]

    # Veri setini eğitim ve test setlerine bölün
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # İlk modelin doğruluğunu ölçme
    initial_rf = RandomForestClassifier(random_state=42)
    initial_rf.fit(X_train, y_train)
    initial_y_pred = initial_rf.predict(X_test)
    initial_accuracy = accuracy_score(y_test, initial_y_pred)
    results.append({'iteration': 0, 'accuracy': initial_accuracy})
    print(f"Initial accuracy: {initial_accuracy:.2f}")

    # Hiperparametre aralığını daraltarak iteratif olarak Grid Search
    for i in range(max_iterations):
        print(f"Iteration {i+1}")
        # Random Forest modelini tanımla
        rf = RandomForestClassifier(random_state=42)
        # Grid Search ile en iyi hiperparametreleri bulma
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise')
        grid_search.fit(X_train, y_train)
        # En iyi modeli ve parametreleri al
        current_model = grid_search.best_estimator_
        current_params = grid_search.best_params_
        y_pred = current_model.predict(X_test)
        current_accuracy = accuracy_score(y_test, y_pred)
        # Sonuçları kaydet
        results.append({
            'iteration': i + 1,
            'params': current_params,
            'accuracy': current_accuracy
        })
        print(f"Best params for iteration {i+1}: {current_params}")
        print(f"Accuracy for iteration {i+1}: {current_accuracy:.2f}")

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = current_model
            best_params = current_params
            # Hiperparametre aralıklarını daralt
            param_grid = {
                'n_estimators': [max(1, current_params['n_estimators'] - 100), current_params['n_estimators'], current_params['n_estimators'] + 100],
                'max_depth': [max(1, current_params['max_depth'] - 5), current_params['max_depth'], current_params['max_depth'] + 5],
                'min_samples_split': [max(2, current_params['min_samples_split'] - 1), current_params['min_samples_split'], current_params['min_samples_split'] + 1],
                'min_samples_leaf': [max(1, current_params['min_samples_leaf'] - 1), current_params['min_samples_leaf'], current_params['min_samples_leaf'] + 1]
            }
        else:
            print("No improvement in this iteration.")
            break

    # Doğruluk skorlarını görselleştirme
    iterations = [result['iteration'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title(f'Grid Search Iterations for {target}')
    plt.grid(True)
    plt.show()

    return best_model, best_params, results

# Age modeli için
age_data_path = 'csv_files\\Age_sayısal_etiketlenmis_veri.csv'
best_age_model, best_age_params, age_results = grid_search_optimization(age_data_path,'Age')
