import numpy as np
from KohonenNetwork import KohonenNetwork
from data_helper import load_and_standardize_europe_data

def main():
    
    print("Red de Kohonen - Clustering de Países Europeos")
    print("-" * 50)
    
    # 1. Cargar datos
    print("Cargando datos...")
    X_standardized, countries, means, stds = load_and_standardize_europe_data('europe.csv')
    print(f"Datos: {len(countries)} países, {X_standardized.shape[1]} características")
    
    # 2. Crear red
    print("Creando red de Kohonen...")
    som = KohonenNetwork(k=5, learning_rate=1.0, columns=7, neighborhood_ratio=3, data=X_standardized)
    print("Red creada con pesos inicializados usando muestras de los datos")
    
    # 3. Entrenar
    print("Entrenando...")
    som.train(X_standardized, epochs=500)
    
    # 4. Clustering
    print("\nResultados:")
    clusters = {}
    
    for i, country in enumerate(countries):
        cluster_pos = som.winner_neuron(X_standardized[i])
        cluster_key = f"({cluster_pos[0]}, {cluster_pos[1]})"
        
        if cluster_key not in clusters:
            clusters[cluster_key] = []
        clusters[cluster_key].append(country)
    
    # 5. Mostrar resultados
    for cluster_pos, country_list in sorted(clusters.items()):
        print(f"\nCluster {cluster_pos}:")
        for country in country_list:
            print(f"  - {country}")
    
    print(f"\nTotal clusters: {len(clusters)}")

if __name__ == "__main__":
    main()
