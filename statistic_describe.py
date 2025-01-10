import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(file_path):
    """Wczytaj dane z pliku CSV."""
    try:
        data = pd.read_csv(file_path)
        print(f"Dane załadowano poprawnie z pliku: {file_path}")
        return data
    except Exception as e:
        print(f"Błąd wczytywania danych: {e}")
        return None

def calculate_numeric_statistics(data):
    """Oblicz statystyki opisowe dla danych liczbowych."""
    print("\n### Statystyki liczbowe ###")
    numeric_stats = data.describe()
    print(numeric_stats)

def calculate_categorical_statistics(data):
    """Oblicz rozkład wartości dla kolumn kategorycznych."""
    print("\n### Rozkład kategorii ###")
    for col in data.select_dtypes(include=['object', 'category']):
        print(f"\nKolumna: {col}")
        print(data[col].value_counts(normalize=True) * 100)

def plot_histograms_individually(data):
    """Generuj osobne histogramy dla każdej kolumny liczbowej."""
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        data[col].hist(bins=30, edgecolor='black')
        plt.title(f'Histogram dla kolumny "{col}"', fontsize=16)
        plt.xlabel(col)
        plt.ylabel('Liczba wystąpień')
        plt.grid(False)
        plt.show()

def plot_categorical_distribution(data, categorical_column):
    """Generuj wykres rozkładu dla kolumny kategorycznej."""
    if categorical_column in data.columns:
        print(f"\n### Rozkład danych dla kolumny '{categorical_column}' ###")
        sns.countplot(y=data[categorical_column], order=data[categorical_column].value_counts().index)
        plt.title(f'Rozkład wartości w kolumnie "{categorical_column}"')
        plt.xlabel('Liczba wystąpień')
        plt.ylabel(categorical_column)
        plt.show()

def calculate_correlation(data):
    """Oblicz macierz korelacji i wyświetl jako heatmapę."""
    print("\n### Macierz korelacji ###")
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Macierz korelacji')
    plt.show()

def calculate_class_balance(data, class_column):
    """Sprawdź liczebność i proporcje klas."""
    if class_column in data.columns:
        print("\n### Rozkład danych w kolumnie klasowej ###")
        print(data[class_column].value_counts())
        print("\n### Proporcje klas ###")
        print(data[class_column].value_counts(normalize=True) * 100)

def main():
    file_path = 'input/RT_IOT2022.csv'  # Ścieżka do pliku
    class_column = 'Attack_type'  # Kolumna wskazująca na typ ataku
    data = load_data(file_path)

    if data is not None:
        calculate_numeric_statistics(data)           # Punkt 1
        calculate_categorical_statistics(data)       # Punkt 2
        plot_histograms_individually(data)           # Punkt 3 (osobne histogramy)
        plot_categorical_distribution(data, class_column)  # Punkt 4
        calculate_correlation(data)                 # Punkt 5
        calculate_class_balance(data, class_column) # Punkt 6

if __name__ == "__main__":
    main()
