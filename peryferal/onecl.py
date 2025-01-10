import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from openpyxl import Workbook
from joblib import dump

# Ścieżki do zapisania wyników i modelu
output_excel_path = r"F:\iot_data\rt-iot2022\output\model_results.xlsx"
model_path = r"F:\iot_data\rt-iot2022\models\RandomForest_model.joblib"

# Utwórz katalog na model, jeśli nie istnieje
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Wczytaj dane
df = pd.read_csv(r'F:\iot_data\rt-iot2022\input\RT_IOT2022.csv')  # Ścieżka do pliku z danymi

# Zakładam, że ostatnia kolumna to target
X = df.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej jako cechy
y = df.iloc[:, -1]   # Ostatnia kolumna jako etykiety

# Wykrywanie kolumn kategorycznych i numerycznych
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing: Standaryzacja i One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Definicja klasyfikatora
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Pipeline: preprocessing + classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu
pipeline.fit(X_train, y_train)

# Zapis przeszkolonego modelu do pliku
dump(pipeline, model_path)
print(f"Model zapisano do {model_path}")

# Predykcja na zbiorze testowym
y_pred = pipeline.predict(X_test)

# Obliczanie metryk
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Wyniki walidacji krzyżowej
cross_val_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')
cross_val_mean = np.mean(cross_val_scores)

# Zapis wyników do Excel
wb = Workbook()
ws = wb.active
ws.title = "Model Metrics"

# Nagłówki tabeli
ws.append(['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cross Validation Mean Accuracy'])

# Dane tabeli
ws.append(['Random Forest', accuracy, precision, recall, f1, cross_val_mean])

# Zapis pliku Excel
wb.save(output_excel_path)
print(f"Wyniki zapisane do {output_excel_path}")
