"""
Implementación de PCA usando la regla de Oja
Calcula la primera componente principal del dataset europe.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from activation_functions import ActivationFunction
from gradient_optimizer import OjaRuleOptimizer
from perceptron_model import SingleLayerPerceptronModel
from training_config import PerceptronTrainingConfig


def load_and_preprocess_data(csv_path):
    """
    Cargar y preprocesar los datos del archivo CSV
    """
    # Cargar datos
    df = pd.read_csv(csv_path)
    
    # Obtener nombres de países
    countries = df['Country'].values
    
    # Seleccionar solo las columnas numéricas (excluir 'Country')
    numeric_columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data = df[numeric_columns].values
    
    # Normalizar los datos (importante para PCA)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data) # x_normalizado = (x - μ) / σ
    
    # Agregar columna de bias
    bias_column = np.ones((normalized_data.shape[0], 1))
    input_data = np.concatenate([bias_column, normalized_data], axis=1)
    
    return input_data, normalized_data, numeric_columns, countries, scaler


def train_oja_pca(input_data, learning_rate=0.01, max_epochs=1000):
    """
    Entrenar el modelo Oja para encontrar la primera componente principal
    """
    # Inicializar configuración de entrenamiento
    PerceptronTrainingConfig(epsilon=1e-5, seed=42, maxEpochs=max_epochs)
    
    # Crear función de activación lineal (necesaria para PCA)
    activation_function = ActivationFunction.create_activation_function("LINEAR") ##w0*1 + w1*x1 + w2*x2 + w3*x3
    
    # Crear optimizador con regla de Oja
    oja_optimizer = OjaRuleOptimizer({"rate": learning_rate})
    
    # Inicializar pesos aleatorios pequeños
    feature_count = input_data.shape[1]
    np.random.seed(42)  # Para reproducibilidad
    initial_weights = np.random.normal(0, 0.1, feature_count)
    
    # Crear modelo
    model = SingleLayerPerceptronModel(activation_function, oja_optimizer, initial_weights)
    
    print("Iniciando entrenamiento con regla de Oja...")
    print(f"Tasa de aprendizaje: {learning_rate}")
    print(f"Máximo de épocas: {max_epochs}")
    print(f"Número de características: {feature_count}")
    print("-" * 50)
    
    # Entrenar usando el método específico para Oja
    model.train_oja_pca(input_data, max_epochs=max_epochs)
    
    return model


def create_pca_bar_chart(model, normalized_data, countries):
    """
    Crear gráfico de barras verticales con los valores de PCA1 por país
    Ordenado de menor a mayor valor PCA1
    """
    # Obtener la primera componente principal (excluir bias)
    oja_component = model.weights[1:]  # Excluir el bias
    
    # Normalizar la componente principal
    oja_component_normalized = oja_component / np.linalg.norm(oja_component)
    
    # Proyectar los datos en la primera componente principal
    pca1_values = np.dot(normalized_data, oja_component_normalized)
    
    # Ordenar países y valores PCA1 de menor a mayor
    sorted_indices = np.argsort(pca1_values)
    sorted_countries = [countries[i] for i in sorted_indices]
    sorted_pca1_values = pca1_values[sorted_indices]
    
    # Crear el gráfico de barras
    plt.figure(figsize=(16, 8))
    
    # Crear barras
    bars = plt.bar(range(len(sorted_countries)), sorted_pca1_values, color='blue', alpha=0.8)
    
    # Configurar el gráfico
    plt.title('PCA1 per Country', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('PCA1', fontsize=12)
    
    # Configurar etiquetas del eje X
    plt.xticks(range(len(sorted_countries)), sorted_countries, rotation=90, ha='right')
    
    # Agregar línea en y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Agregar grid horizontal
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Ajustar layout para que las etiquetas no se corten
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig('pca1_per_country.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return oja_component_normalized, pca1_values, sorted_countries, sorted_pca1_values


def print_results(oja_component, feature_names, pca1_values, countries):
    """
    Imprimir resultados detallados
    """
    print("\n" + "="*60)
    print("RESULTADOS DE PCA CON REGLA DE OJA")
    print("="*60)
    
    print("\nPrimera componente principal (Oja):")
    for i, (feature, weight) in enumerate(zip(feature_names, oja_component)):
        print(f"  {feature}: {weight:.6f}")
    
    print(f"\nNorma de la componente Oja: {np.linalg.norm(oja_component):.6f}")
    
    # Imprimir la fórmula Y1
    print("\n" + "="*60)
    print("FÓRMULA Y1 (PRIMERA COMPONENTE PRINCIPAL)")
    print("="*60)
    
    # Construir la fórmula
    formula_parts = []
    for feature, weight in zip(feature_names, oja_component):
        if weight >= 0:
            formula_parts.append(f"{weight:.4f}{feature}")
        else:
            formula_parts.append(f"{weight:.4f}{feature}")
    
    formula = " + ".join(formula_parts)
    # Reemplazar "+ -" con "- " para una mejor presentación
    formula = formula.replace("+ -", "- ")
    
    print(f"\nY1 = {formula}")
    print("\n" + "="*60)
    
    # Mostrar las características más importantes
    print("\nCaracterísticas más importantes (por valor absoluto):")
    abs_weights = np.abs(oja_component)
    sorted_indices = np.argsort(abs_weights)[::-1]
    
    for i, idx in enumerate(sorted_indices):
        print(f"  {i+1}. {feature_names[idx]}: {oja_component[idx]:.6f}")
    
    # Mostrar países con valores PCA1 más altos y más bajos
    print("\nPaíses con valores PCA1 más altos:")
    sorted_countries_high = sorted(zip(countries, pca1_values), key=lambda x: x[1], reverse=True)
    for i, (country, value) in enumerate(sorted_countries_high[:5]):
        print(f"  {i+1}. {country}: {value:.4f}")
    
    print("\nPaíses con valores PCA1 más bajos:")
    for i, (country, value) in enumerate(sorted_countries_high[-5:]):
        print(f"  {i+1}. {country}: {value:.4f}")


def main():
    """
    Función principal
    """
    print("Implementación de PCA usando la Regla de Oja")
    print("Dataset: europe.csv")
    print("="*50)
    
    # Cargar y preprocesar datos
    csv_path = "europe.csv"
    input_data, normalized_data, feature_names, countries, scaler = load_and_preprocess_data(csv_path)
    
    print(f"Datos cargados: {input_data.shape[0]} muestras, {len(feature_names)} características")
    print(f"Características: {', '.join(feature_names)}")
    print(f"Países: {len(countries)} países europeos")
    
    # Entrenar modelo Oja
    model = train_oja_pca(input_data, learning_rate=0.01, max_epochs=1000)
    
    # Crear gráfico de barras
    oja_component, pca1_values, sorted_countries, sorted_pca1_values = create_pca_bar_chart(model, normalized_data, countries)
    
    # Imprimir resultados
    print_results(oja_component, feature_names, pca1_values, countries)
    
    print("\n" + "="*60)
    print("Análisis completado. Gráfico guardado en 'pca1_per_country.png'")
    print("="*60)


if __name__ == "__main__":
    main()
