import os
import numpy as np
import pandas as pd

# Sklearn / Imblearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Plotly do wizualizacji (opcjonalnie)
import plotly.express as px
import plotly.figure_factory as ff  # <-- potrzebne do wykresu macierzy pomyłek

# Zapis do Excel i modelu
from openpyxl import Workbook
from joblib import dump

# =============================
# 1. Ścieżki do zapisania wyników
# =============================
output_excel_path = r"F:\iot_data\rt-iot2022\output\model_results.xlsx"
model_path = r"F:\iot_data\rt-iot2022\models\RandomForest_model.joblib"

os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# =========================================
# 2. Wczytanie danych
# =========================================
df = pd.read_csv(r'F:\iot_data\rt-iot2022\input\RT_IOT2022.csv')
print("Dane zostały wczytane pomyślnie.")
print(f"Rozmiar danych: {df.shape}")

# =========================================
# 3. Tworzenie kolumny Label (0/1)
# =========================================
normal_classes = ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb']
df['Label'] = df['Attack_type'].apply(lambda x: 0 if x in normal_classes else 1)

# =========================================
# 4. Sprawdzenie rozkładu
# =========================================
print("\nRozkład etykiety (Label) w całym zbiorze:")
print(df['Label'].value_counts())

fig = px.histogram(
    df, x='Label',
    title='Rozkład etykiety (0=normal, 1=atak) - CAŁY zbiór',
    color='Label'
)
fig.show()

# =========================================
# 5. Definiujemy cechy (features)
# =========================================
features = [
    'flow_duration',
    'fwd_pkts_tot',
    'bwd_pkts_tot',
    'fwd_data_pkts_tot',
    'bwd_data_pkts_tot',
    'fwd_pkts_per_sec',
    'bwd_pkts_per_sec',
    'flow_pkts_per_sec',
    'down_up_ratio',
    'fwd_header_size_tot',
    'bwd_header_size_tot',
    # Jeżeli masz kolumny kategoryczne (np. 'proto', 'service'),
    # dopisz je tutaj – wtedy OneHotEncoder będzie miał co kodować.
]

X = df[features]
y = df['Label']

# =========================================
# 6. Określenie kolumn kategorycznych i numerycznych
# =========================================
numeric_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

print("\nKolumny numeryczne:", list(numeric_cols))
print("Kolumny kategoryczne:", list(cat_cols))

# =========================================
# 7. Budowa ColumnTransformer (preprocessing)
# =========================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# =========================================
# 8. Budowa Pipeline z imblearn:
#    preprocessor -> SMOTE -> RandomForest
# =========================================
pipeline = ImbPipeline([
    ('preprocessing', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        bootstrap=True
    ))
])

# =========================================
# 9. Podział na zbiór treningowy i testowy
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 10. Cross-validation na zbiorze treningowym
# =========================================
scores = cross_val_score(
    pipeline, 
    X_train,  
    y_train, 
    cv=10, 
    scoring='accuracy'
)
cv_mean = scores.mean()

print("\n=== CROSS-VALIDATION (10-fold) na zbiorze treningowym ===")
print("Wyniki poszczególnych foldów (accuracy):", scores)
print("Średnia (accuracy):", cv_mean)

# =========================================
# 11. Trenowanie Pipeline na całym zbiorze treningowym
# =========================================
pipeline.fit(X_train, y_train)

# =========================================
# 12. Ewaluacja końcowa na zbiorze testowym
# =========================================
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== EWALUACJA KOŃCOWA (zbiór testowy) ===")
print("Dokładność (Accuracy):", accuracy)
print("Precyzja (Precision):", precision)
print("Czułość (Recall):", recall)
print("Wynik F1:", f1)

# =========================================
# 13. Dodatkowa analiza: dysproporcja, macierz pomyłek, AUC
# =========================================
print("\n--- Dysproporcja w zbiorze testowym ---")
print(y_test.value_counts())

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
print("\n--- Macierz pomyłek (Confusion Matrix) ---")
print(cm)

y_prob = pipeline.predict_proba(X_test)[:, 1]
auc_value = roc_auc_score(y_test, y_prob)
print(f"\n--- AUC (Area Under ROC Curve) = {auc_value:.4f}")

# ====== WIZUALIZACJA MACIERZY POMYŁEK (HEATMAP) ======
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted: Normal (0)", "Predicted: Attack (1)"],
    y=["Actual: Normal (0)", "Actual: Attack (1)"],
    colorscale='Blues',
    showscale=True
)
fig_cm.update_layout(
    title="Confusion Matrix (Plotly Heatmap)",
    xaxis=dict(title="Predicted Label"),
    yaxis=dict(title="True Label")
)
fig_cm.show()

# =========================================
# 14. Zapis wyników do pliku Excel
# =========================================
wb = Workbook()
ws = wb.active
ws.title = "Model Metrics"

# Nagłówki
ws.append(["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Mean Accuracy", "AUC"])
ws.append([
    "RandomForest",
    accuracy,
    precision,
    recall,
    f1,
    cv_mean,
    auc_value
])

wb.save(output_excel_path)
print(f"\nWyniki zapisano do pliku Excel: {output_excel_path}")

# =========================================
# 15. Zapis Pipeline (modelu) do .joblib
# =========================================
dump(pipeline, model_path)
print(f"Pipeline zapisano do: {model_path}")
