import pandas as pd
# Ścieżka do pliku
file_path = "input/RT_IOT2022.csv"

# Wczytanie pliku
data = pd.read_csv(file_path)

# Pobranie dynamicznej listy kolumn
columns = list(data.columns)

# Wyświetlenie nazw kolumn
print("Nazwy kolumn w danych:")
for i, col in enumerate(columns, start=1):
    print(f"{i}. {col}")

# Wybór kolumny
selected_column = input("\nPodaj nazwę kolumny, której zawartość chcesz zobaczyć: ")

# Sprawdzenie, czy kolumna istnieje
if selected_column in data.columns:
    # Drukowanie całej zawartości kolumny
    print(f"\nZawartość kolumny '{selected_column}' (cała zawartość):")
    # print(data[selected_column].to_string(index=False))  # Zakomentuj tę linię, jeśli nie chcesz wyświetlać całej zawartości

    # Drukowanie unikalnych wartości
    print(f"\nUnikalne wartości w kolumnie '{selected_column}':")
    print(data[selected_column].unique())
else:
    print("Podana kolumna nie istnieje w danych.")
