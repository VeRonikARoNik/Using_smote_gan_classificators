import os
import numpy as np
import pandas as pd

# scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier

# PyTorch do GAN-a
import torch
import torch.nn as nn
import torch.optim as optim

# Zapis modelu, Excel, wykresy
from openpyxl import Workbook
from joblib import dump
import plotly.express as px
import plotly.figure_factory as ff

# ---------------------------------------------------------
# Parametry, urządzenie i klasy sieci (Generator, Discriminator)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 16      # wymiar szumu (wejście do Generatora)
lr = 0.0002          # learning rate
batch_size = 64
epochs = 200

criterion = nn.BCELoss()

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 1. Wczytanie danych
# ---------------------------------------------------------
df = pd.read_csv(r'F:\iot_data\rt-iot2022\input\RT_IOT2022.csv')
print("Dane wczytane, kształt:", df.shape)

# ---------------------------------------------------------
# 2. Definiujemy klasy normalne i etykietę
# ---------------------------------------------------------
normal_classes = ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb']
df['Label'] = df['Attack_type'].apply(lambda x: 0 if x in normal_classes else 1)
count_0 = (df['Label'] == 0).sum()
count_1 = (df['Label'] == 1).sum()
print(f"Liczba próbek klasy 0: {count_0}")
print(f"Liczba próbek klasy 1: {count_1}")

# ---------------------------------------------------------
# 3. Kodowanie kolumn kategorycznych i wybór cech
# ---------------------------------------------------------
encoder_proto = LabelEncoder()
encoder_service = LabelEncoder()

df['proto_enc'] = encoder_proto.fit_transform(df['proto'].astype(str))
df['service_enc'] = encoder_service.fit_transform(df['service'].astype(str))

# Zostawiamy TYLKO cechy z base_features + proto_enc + service_enc:
base_features = [
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

# (Brak extra_features — całkowicie usunięte)
all_features = base_features + ['proto_enc', 'service_enc']

# Upewniamy się, że kolumny istnieją w DataFrame
for col in all_features:
    if col not in df.columns:
        print(f"[UWAGA] Kolumna '{col}' nie istnieje!")

X = df[all_features].copy()
y = df['Label'].copy()
print("Features used in the model:", all_features)
print(df[all_features].head())
print(df['proto_enc'].value_counts())
print(df['service_enc'].value_counts())
# ---------------------------------------------------------
# 4. Podział na train i test
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 5. Walidacja krzyżowa (10-krotna) z oversamplingiem przy pomocy GAN
# ---------------------------------------------------------
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []
auc_scores = []

# Opcja: zapisywanie confusion matrix z każdego folda (jeśli chcesz potem je pokazać)
cm_list = []

# Konwersja do numpy
X_train_np = X_train.values
y_train_np = y_train.values

fold_idx = 1

for train_idx, val_idx in skf.split(X_train_np, y_train_np):
    print(f"\n=== FOLD {fold_idx} / {n_splits} ===")
    fold_idx += 1
    
    X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
    y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]
    
    # Skalowanie w tym foldzie
    scaler_fold = StandardScaler()
    X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
    X_val_fold_scaled   = scaler_fold.transform(X_val_fold)
    
    # Wyznaczamy klasę mniejszościową (założenie: 0 jest mniejszościowe)
    minority_class = 0  
    X_fold_min = X_train_fold_scaled[y_train_fold == minority_class]
    X_fold_maj = X_train_fold_scaled[y_train_fold != minority_class]
    
    needed = X_fold_maj.shape[0] - X_fold_min.shape[0]
    
    # Definiujemy osobne egzemplarze Generatora/Dyskryminatora dla tego folda
    generator_fold = Generator(latent_dim, X_fold_min.shape[1]).to(device)
    discriminator_fold = Discriminator(X_fold_min.shape[1]).to(device)
    
    optim_G_fold = optim.Adam(generator_fold.parameters(), lr=lr)
    optim_D_fold = optim.Adam(discriminator_fold.parameters(), lr=lr)
    
    # Przygotowujemy tensor dla klasy mniejszościowej
    minority_data_fold = torch.tensor(X_fold_min, dtype=torch.float).to(device)
    
    # Trenujemy GAN (np. 200 epok)
    for epoch_cv in range(epochs):
        idx_ = np.random.randint(0, minority_data_fold.shape[0], batch_size)
        real_samples_fold = minority_data_fold[idx_]

        z_fold = torch.randn(batch_size, latent_dim).to(device)
        fake_samples_fold = generator_fold(z_fold)

        real_labels_fold = torch.ones(batch_size, 1).to(device)
        fake_labels_fold = torch.zeros(batch_size, 1).to(device)

        real_output_fold = discriminator_fold(real_samples_fold)
        fake_output_fold = discriminator_fold(fake_samples_fold.detach())

        d_loss_real_fold = criterion(real_output_fold, real_labels_fold)
        d_loss_fake_fold = criterion(fake_output_fold, fake_labels_fold)
        d_loss_fold = d_loss_real_fold + d_loss_fake_fold
        
        optim_D_fold.zero_grad()
        d_loss_fold.backward()
        optim_D_fold.step()

        z_fold = torch.randn(batch_size, latent_dim).to(device)
        fake_samples_fold = generator_fold(z_fold)
        g_output_fold = discriminator_fold(fake_samples_fold)
        g_loss_fold = criterion(g_output_fold, real_labels_fold)

        optim_G_fold.zero_grad()
        g_loss_fold.backward()
        optim_G_fold.step()

    # Po wytrenowaniu – generujemy potrzebne próbki (jeśli needed > 0)
    if needed > 0:
        fake_data_list_fold = []
        while len(fake_data_list_fold) < needed:
            z_fold = torch.randn(batch_size, latent_dim).to(device)
            fake_data_fold = generator_fold(z_fold).detach().cpu().numpy()
            fake_data_list_fold.append(fake_data_fold)
        fake_data_array_fold = np.vstack(fake_data_list_fold)[:needed]
        
        X_oversampled_min_fold = np.vstack([X_fold_min, fake_data_array_fold])
        y_oversampled_min_fold = np.full(X_oversampled_min_fold.shape[0], minority_class, dtype=int)
        
        # Druga klasa
        X_fold_maj_ = X_fold_maj
        y_fold_maj_ = np.full(X_fold_maj.shape[0], 1 - minority_class, dtype=int)
        
        X_train_fold_final = np.vstack([X_fold_maj_, X_oversampled_min_fold])
        y_train_fold_final = np.concatenate([y_fold_maj_, y_oversampled_min_fold])
    else:
        X_train_fold_final = np.vstack([X_fold_maj, X_fold_min])
        y_train_fold_final = np.concatenate([
            np.ones(X_fold_maj.shape[0], dtype=int),
            np.zeros(X_fold_min.shape[0], dtype=int)
        ])
    
    # Trenujemy klasyfikator
    clf_fold = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        bootstrap=True
    )
    clf_fold.fit(X_train_fold_final, y_train_fold_final)
    
    # Ewaluacja
    y_val_pred_fold = clf_fold.predict(X_val_fold_scaled)
    y_val_prob_fold = clf_fold.predict_proba(X_val_fold_scaled)[:, 1]
    
    acc_fold = accuracy_score(y_val_fold, y_val_pred_fold)
    prec_fold = precision_score(y_val_fold, y_val_pred_fold)
    rec_fold  = recall_score(y_val_fold, y_val_pred_fold)
    f1_fold   = f1_score(y_val_fold, y_val_pred_fold)
    auc_fold  = roc_auc_score(y_val_fold, y_val_prob_fold)
    
    acc_scores.append(acc_fold)
    prec_scores.append(prec_fold)
    rec_scores.append(rec_fold)
    f1_scores.append(f1_fold)
    auc_scores.append(auc_fold)
    
    # (Opcjonalnie) Zapisujemy confusion_matrix z tego folda
    cm_fold = confusion_matrix(y_val_fold, y_val_pred_fold)
    cm_list.append(cm_fold)
    
    print(f"  Fold Accuracy:  {acc_fold:.4f}, Precision: {prec_fold:.4f}, Recall: {rec_fold:.4f}, "
          f"F1: {f1_fold:.4f}, AUC: {auc_fold:.4f}")

# Podsumowanie 10-krotnej walidacji
print("\n=== WYNIKI 10-krotnej WALIDACJI KRZYŻOWEJ (na zbiorze treningowym) ===")
print(f"Średnia Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"Średnia Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
print(f"Średnia Recall:    {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
print(f"Średnia F1:        {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Średnia AUC:       {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

# ---------------------------------------------------------
# 5a. Wykresy CV – metryki vs. nr folda
# ---------------------------------------------------------
cv_results_df = pd.DataFrame({
    'fold': list(range(1, n_splits+1)),
    'Accuracy': acc_scores,
    'Precision': prec_scores,
    'Recall': rec_scores,
    'F1': f1_scores,
    'AUC': auc_scores
})

# Przykład 1: line chart
fig_cv_line = px.line(
    cv_results_df, 
    x='fold', 
    y=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
    markers=True,
    title='Wyniki walidacji krzyżowej - metryki vs. fold'
)
fig_cv_line.update_layout(yaxis=dict(range=[0.0, 1.05]))  # żeby skala y kończyła się np. na 1.05
fig_cv_line.show()

# Przykład 2: bar chart (możesz wybrać jedną metrykę, np. Accuracy)
fig_cv_bar = px.bar(
    cv_results_df,
    x='fold',
    y='Accuracy',
    title='Accuracy w poszczególnych foldach'
)
fig_cv_bar.update_layout(yaxis=dict(range=[0.0, 1.05]))
fig_cv_bar.show()

# ---------------------------------------------------------
# 5b. Macierze konfuzji z cross-validation
# ---------------------------------------------------------
# (A) Uśredniona (suma) macierz konfuzji
sum_cm = np.sum(cm_list, axis=0)  # sumuje macierze z każdego folda
fig_cm_sum = ff.create_annotated_heatmap(
    z=sum_cm,
    x=['Predicted 0', 'Predicted 1'],
    y=['True 0', 'True 1'],
    colorscale='Blues'
)
fig_cm_sum.update_layout(title='Suma macierzy konfuzji ze wszystkich foldów (CV)')
fig_cm_sum.show()

# (B) Wyświetlanie macierzy z każdego folda osobno (jeśli chcesz)
# for i, cm_fold in enumerate(cm_list):
#     fig_cm_fold = ff.create_annotated_heatmap(
#         z=cm_fold,
#         x=['Predicted 0', 'Predicted 1'],
#         y=['True 0', 'True 1'],
#         colorscale='Blues'
#     )
#     fig_cm_fold.update_layout(title=f'Macierz konfuzji - Fold {i+1}')
#     fig_cm_fold.show()

# ---------------------------------------------------------
# 6. Wyświetlenie rozkładu PRZED oversamplingiem (tylko train)
# ---------------------------------------------------------
train_df_before = pd.DataFrame({'Label': y_train})
fig_before = px.histogram(
    train_df_before, 
    x='Label', 
    title='Rozkład etykiety (0=normal, 1=atak) - PRZED oversamplingiem (Train)',
    color='Label'
)
fig_before.show()

# ---------------------------------------------------------
# 7. Skalowanie (na całym train i test)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Sprawdzamy, która klasa jest faktycznie mniejszościowa
minority_class = 0  # Założenie, że '0' jest mniejszością
X_train_min = X_train_scaled[y_train == minority_class]
X_train_maj = X_train_scaled[y_train != minority_class]

print(f"\nLiczba próbek 'minority_class' = {minority_class}: {X_train_min.shape[0]}")
print(f"Liczba próbek 'majority_class': {X_train_maj.shape[0]}")

# ---------------------------------------------------------
# 8. Definiujemy Generator i Discriminator (nowy egzemplarz do finalnego oversamplingu)
# ---------------------------------------------------------
generator = Generator(latent_dim, X_train_min.shape[1]).to(device)
discriminator = Discriminator(X_train_min.shape[1]).to(device)

optim_G = optim.Adam(generator.parameters(), lr=lr)
optim_D = optim.Adam(discriminator.parameters(), lr=lr)

minority_data = torch.tensor(X_train_min, dtype=torch.float).to(device)

# ---------------------------------------------------------
# 9. Trening GAN (finalne dopasowanie na całym X_train_min)
# ---------------------------------------------------------
for epoch in range(epochs):
    idx = np.random.randint(0, minority_data.shape[0], batch_size)
    real_samples = minority_data[idx]

    z = torch.randn(batch_size, latent_dim).to(device)
    fake_samples = generator(z)

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    real_output = discriminator(real_samples)
    fake_output = discriminator(fake_samples.detach())

    d_loss_real = criterion(real_output, real_labels)
    d_loss_fake = criterion(fake_output, fake_labels)
    d_loss = d_loss_real + d_loss_fake

    optim_D.zero_grad()
    d_loss.backward()
    optim_D.step()

    # Generator
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_samples = generator(z)
    g_output = discriminator(fake_samples)
    g_loss = criterion(g_output, real_labels)

    optim_G.zero_grad()
    g_loss.backward()
    optim_G.step()

    if (epoch+1) % 50 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}")

# ---------------------------------------------------------
# 10. Generowanie dodatkowych próbek minority_class
# ---------------------------------------------------------
minor_count = X_train_min.shape[0]
major_count = X_train_maj.shape[0]

needed = major_count - minor_count
if needed <= 0:
    print(f"\nKlasa {minority_class} nie jest faktycznie mniejszościowa lub jest równa. "
          "Nie generujemy dodatkowych próbek.\n")
    X_train_final = np.vstack([X_train_maj, X_train_min])
    y_train_final = np.concatenate([
        np.ones(X_train_maj.shape[0], dtype=int),
        np.zeros(X_train_min.shape[0], dtype=int)
    ])
else:
    print(f"\nGenerujemy {needed} próbek klasy {minority_class} za pomocą GAN...")
    fake_data_list = []
    while len(fake_data_list) < needed:
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(z).detach().cpu().numpy()
        fake_data_list.append(fake_data)
    fake_data_array = np.vstack(fake_data_list)[:needed]

    X_oversampled_min = np.vstack([X_train_min, fake_data_array])
    y_oversampled_min = np.full(X_oversampled_min.shape[0], minority_class, dtype=int)

    X_train_maj_ = X_train_maj
    y_train_maj_ = np.full(X_train_maj_.shape[0], 1 - minority_class, dtype=int)

    X_train_final = np.vstack([X_train_maj_, X_oversampled_min])
    y_train_final = np.concatenate([y_train_maj_, y_oversampled_min])

# ---------------------------------------------------------
# 11. Wyświetlenie rozkładu PO oversamplingu
# ---------------------------------------------------------
train_df_after = pd.DataFrame({'Label': y_train_final})
fig_after = px.histogram(
    train_df_after,
    x='Label',
    title='Rozkład etykiety (0=normal, 1=atak) - PO oversamplingu (Train)',
    color='Label'
)
fig_after.show()

# ---------------------------------------------------------
# 12. Trenowanie klasyfikatora (RandomForest)
# ---------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    bootstrap=True
)
clf.fit(X_train_final, y_train_final)

# ---------------------------------------------------------
# 13. Ewaluacja na zbiorze testowym
# ---------------------------------------------------------
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)

print("\n=== WYNIKI EWALUACJI (TEST) ===")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]
auc_val = roc_auc_score(y_test, y_prob)
print("AUC:", auc_val)

# ------------------------------
# 13a. Wizualizacja macierzy konfuzji testu (Plotly)
# ------------------------------
fig_cm_test = ff.create_annotated_heatmap(
    z=cm,
    x=['Predicted 0', 'Predicted 1'],
    y=['True 0', 'True 1'],
    colorscale='Blues'
)
fig_cm_test.update_layout(title='Macierz konfuzji na zbiorze testowym')
fig_cm_test.show()

# ---------------------------------------------------------
# 14. Zapis metryk do Excela
# ---------------------------------------------------------
wb = Workbook()
ws = wb.active
ws.title = "Model Metrics"
ws.append(["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
ws.append(["RandomForest+GAN", acc, prec, rec, f1, auc_val])

output_excel_path = r"F:\iot_data\rt-iot2022\output\model_results_gan.xlsx"
wb.save(output_excel_path)
print(f"\nWyniki zapisano do: {output_excel_path}")

# ---------------------------------------------------------
# 15. Zapis modelu
# ---------------------------------------------------------
model_path = r"F:\iot_data\rt-iot2022\models\RandomForest_model_gan.joblib"
dump(clf, model_path)
print(f"Model zapisano do: {model_path}")
