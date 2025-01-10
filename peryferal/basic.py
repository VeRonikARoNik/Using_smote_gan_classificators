
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Wczytanie danych
df = pd.read_csv('input/RT_IOT2022.csv')  # Zmień nazwę pliku, jeśli jest inna
print("Dane zostały wczytane pomyślnie.")
print(f"Rozmiar danych: {df.shape}")

# Przekształcenie do problemu dwuklasowego
target_classes = ['MQTT_Publish', 'ARP_poisioning']
df_binary = df[df['Attack_type'].isin(target_classes)].copy()

# Kodowanie etykiet - MQTT jako 0 (normalny ruch), ARP_poisioning jako 1 (atak)
le = LabelEncoder()
df_binary['Label'] = le.fit_transform(df_binary['Attack_type'])
if le.classes_[0] != 'MQTT_Publish':
    df_binary['Label'] = df_binary['Label'].apply(lambda x: 1 - x)  # Odwróć wartości, jeśli potrzeba

# Sprawdzenie rozkładu klas po filtracji
print("Rozkład wybranych klas przed SMOTE:")
print(df_binary['Attack_type'].value_counts())

# Wizualizacja rozkładu klas przed SMOTE
fig = px.histogram(df_binary, x='Attack_type', title='Rozkład klas Attack_type przed SMOTE', 
                   labels={'Attack_type': 'Typ Ruchu'}, color='Attack_type')
fig.show()

# Wybór cech
features = ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot']  # Przykładowe cechy
X = df_binary[features]
y = df_binary['Label']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Zastosowanie SMOTE do zbioru treningowego
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Sprawdzenie rozkładu klas po zastosowaniu SMOTE
print("Rozkład klas po SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Wizualizacja rozkładu klas po SMOTE
fig = px.histogram(y_train_balanced, title='Rozkład klas po SMOTE', 
                   labels={'value': 'Typ Ruchu'}, color=y_train_balanced)
fig.show()

# Trenowanie modelu
model = RandomForestClassifier(
    n_estimators=200,         # Liczba drzew
    max_depth=15,             # Maksymalna głębokość drzewa
    min_samples_split=5,      # Minimalna liczba próbek do podziału
    min_samples_leaf=2,       # Minimalna liczba próbek w liściu
    max_features='sqrt',      # Liczba cech przy każdym podziale
    class_weight='balanced',  # Automatyczne balansowanie wag klas
    random_state=42,          # Ustalona losowość dla powtarzalności
    bootstrap=True            # Użycie próbek bootstrap
)
model.fit(X_train_balanced, y_train_balanced)

# Ewaluacja modelu
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Dokładność:", accuracy)
print("Precyzja:", precision)
print("Czułość:", recall)
print("Wynik F1:", f1)