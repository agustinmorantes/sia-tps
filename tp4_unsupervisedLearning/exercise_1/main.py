import numpy as np
import json
from datetime import datetime
from KohonenNetwork import KohonenNetwork
from data_helper import load_and_standardize_europe_data

def save_clustering_results(clusters, k, som_weights):
    """Guarda los resultados del clustering en un archivo JSON con timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'clustering_results_{timestamp}.json'
    
    results = {
        'clusters': clusters,
        'grid_size': k,
        'total_clusters': len(clusters),
        'timestamp': timestamp,
        'weights': som_weights.tolist()  # Guardar pesos para calcular heatmap
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # También guardar el último resultado como "latest"
    with open('clustering_results_latest.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Resultados guardados en: {filename}")
    print(f"✓ Última versión en: clustering_results_latest.json")
    return filename

def main():
    
    print("Red de Kohonen - Clustering de Países Europeos")
    print("-" * 50)
    
    # 1. Cargar datos
    print("Cargando datos...")
    X_standardized, countries, means, stds = load_and_standardize_europe_data('europe.csv')
    print(f"Datos: {len(countries)} países, {X_standardized.shape[1]} características")
    
    # 2. Crear red
    print("Creando red de Kohonen...")
    som = KohonenNetwork(k=3, learning_rate=1.0, columns=7, neighborhood_ratio=3, data=X_standardized)
    print("Red creada con pesos inicializados usando muestras de los datos")
    
    # 3. Entrenar
    print("Entrenando...")
    som.train(X_standardized, epochs=3500)
    
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
    
    # 6. Guardar resultados (incluyendo pesos reales)
    filename = save_clustering_results(clusters, som.k, som.map)
    
    # 7. Generar visualización
    print("\n" + "="*50)
    print("Para visualizar los resultados, ejecuta:")
    print(f"  python visualize_clusters.py {filename}")
    print("  o simplemente: python visualize_clusters.py")
    print("="*50)

if __name__ == "__main__":
    main()
