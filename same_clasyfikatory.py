import os
import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# XGBoost
from xgboost import XGBClassifier

# Plotly do wizualizacji
import plotly.express as px
import plotly.figure_factory as ff

# Zapis do Excel i modelu
from openpyxl import Workbook
from joblib import dump

# =============================
# 1. Ścieżki do zapisania wyników
# =============================
output_excel_path = r"F:\iot_data\rt-iot2022\output\model_results_nooversampling.xlsx"
model_folder = r"F:\iot_data\rt-iot2022\models"
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

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
# 4. Sprawdzenie rozkładu w całym zbiorze
# =========================================
fig_all = px.histogram(
    df, x='Label',
    title='Rozkład etykiety (0=normal, 1=atak) - CAŁY zbiór',
    color='Label'
)
fig_all.show()

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
    'bwd_header_size_tot'
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
# 8. Podział na zbiór treningowy i testowy
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Wykres rozkładu PRZED (train)
fig_before = px.histogram(
    pd.DataFrame({'Label': y_train}),
    x='Label',
    title='Rozkład etykiety (0=normal, 1=atak) - BEZ oversamplingu (Train)',
    color='Label'
)
fig_before.show()

# =========================================
# 9. Definicja klasyfikatorów (bez SMOTE / ADASYN)
# =========================================
classifiers = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        bootstrap=True
    ),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'XGBClassifier': XGBClassifier(eval_metric='logloss',  # usuń/zmień, by uniknąć ostrzeżeń
                                   random_state=42)
}

# =========================================
# 10. Przygotowanie arkusza Excel z wynikami
# =========================================
wb = Workbook()
ws = wb.active
ws.title = "Model Metrics"
# Nagłówki kolumn w Excelu
ws.append([
    "Classifier", 
    "Train_CV_Accuracy", 
    "Test_Accuracy", 
    "Test_Precision", 
    "Test_Recall", 
    "Test_F1", 
    "Test_AUC"
])

# =========================================
# 11. Pętla po klasyfikatorach
# =========================================
for clf_name, clf_obj in classifiers.items():
    # Budujemy pipeline: preprocessor -> classifier (bez oversamplingu)
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', clf_obj)
    ])
    
    # --- CROSS-VALIDATION (na train) ---
    scores = cross_val_score(
        pipeline, 
        X_train,  
        y_train, 
        cv=5,             # np. 5-fold cross-validation
        scoring='accuracy'
    )
    cv_mean = np.mean(scores)
    
    # --- Trenowanie na całym train ---
    pipeline.fit(X_train, y_train)
    
    # --- Ewaluacja na test ---
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # AUC - potrzebna pred_proba
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc_value = roc_auc_score(y_test, y_prob)
    
    # -- Wypisujemy do konsoli
    print(f"\n=== Classifier: {clf_name} (bez oversamplingu) ===")
    print(f"CV Mean Accuracy (train): {cv_mean:.4f}")
    print(f"TEST -> Acc: {accuracy:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")
    
    # --- Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # --- Zapis macierzy konfuzji do pliku HTML (bez blokowania)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted: 0", "Predicted: 1"],
        y=["Actual: 0", "Actual: 1"],
        colorscale='Blues',
        showscale=True
    )
    fig_cm.update_layout(
        title=f"Confusion Matrix - {clf_name} (bez oversamplingu)",
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label")
    )
    
    cm_filename = f"cm_NOoversampling_{clf_name}.html"
    cm_filepath = os.path.join(model_folder, cm_filename)
    fig_cm.write_html(cm_filepath)
    print(f"Macierz konfuzji zapisana do: {cm_filepath}")
    
    # --- Zapis do EXCELA
    ws.append([
        clf_name,
        cv_mean,
        accuracy,
        precision,
        recall,
        f1,
        auc_value
    ])
    
    # --- Zapis Pipeline (opcjonalnie)
    model_filename = f"{clf_name}_NOoversampling.joblib"
    model_path = os.path.join(model_folder, model_filename)
    dump(pipeline, model_path)
    print(f"Model zapisany do: {model_path}")

# =========================================
# 12. Zapis pliku Excel (wszystkie wyniki)
# =========================================
wb.save(output_excel_path)
print(f"\nWyniki zapisano do pliku Excel: {output_excel_path}")
