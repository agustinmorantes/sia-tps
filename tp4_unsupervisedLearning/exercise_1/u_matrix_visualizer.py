import json
import matplotlib.pyplot as plt
import numpy as np

def load_clustering_results(filename='clustering_results_latest.json'):
    """Carga los resultados del clustering desde un archivo JSON"""
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_u_matrix(weights):
    """
    Calcula la U-Matrix (Unified Distance Matrix) que muestra las distancias
    promedio entre neuronas vecinas en el mapa de Kohonen
    """
    k = weights.shape[0]
    u_matrix = np.zeros((k, k))
    
    for i in range(k):
        for j in range(k):
            distances = []
            # Calcular distancia a vecinos inmediatos (radio = 1)
            neighbors = [
                (i-1, j),  # arriba
                (i+1, j),  # abajo
                (i, j-1),  # izquierda
                (i, j+1)   # derecha
            ]
            
            for ni, nj in neighbors:
                if 0 <= ni < k and 0 <= nj < k:
                    dist = np.linalg.norm(weights[i, j] - weights[ni, nj])
                    distances.append(dist)
            
            # Promedio de distancias a vecinos
            u_matrix[i, j] = np.mean(distances) if distances else 0
    
    return u_matrix

def visualize_u_matrix(results, output_file='u_matrix.png'):
    """
    Visualiza la U-Matrix del mapa de Kohonen
    """
    # Obtener pesos si están disponibles
    weights_data = results.get('weights', None)
    if weights_data is None:
        print("Error: No se encontraron pesos en los resultados.")
        print("Ejecuta main.py primero para generar los pesos.")
        return
    
    weights = np.array(weights_data)
    grid_size = weights.shape[0]
    
    # Calcular U-Matrix
    u_matrix = calculate_u_matrix(weights)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Crear heatmap de la U-Matrix
    im = ax.imshow(u_matrix, cmap='gray', interpolation='nearest')
    
    # Configurar ejes
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
  
    
    # Título
    plt.title('Mapa de distancias promedio entre neuronas', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Añadir valores en cada celda
    for i in range(grid_size):
        for j in range(grid_size):
            # Determinar color del texto basado en el valor de la celda
            # Valores altos = fondos claros = texto negro
            # Valores bajos = fondos oscuros = texto blanco
            cell_value = u_matrix[i, j]
            min_val = u_matrix.min()
            max_val = u_matrix.max()
            
            # Normalizar el valor entre 0 y 1
            normalized_value = (cell_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            # Usar texto blanco para valores bajos (fondos oscuros) y negro para valores altos (fondos claros)
            text_color = 'white' if normalized_value < 0.5 else 'black'
            
            ax.text(j, i, f'{u_matrix[i, j]:.2f}',
                   ha="center", va="center", 
                   color=text_color, fontsize=12, fontweight='bold')
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)

    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ U-Matrix guardado en: {output_file}")
    plt.show()
    
    # Mostrar estadísticas
    print(f"\nEstadísticas de la U-Matrix:")
    print(f"Distancia mínima: {u_matrix.min():.3f}")
    print(f"Distancia máxima: {u_matrix.max():.3f}")
    print(f"Distancia promedio: {u_matrix.mean():.3f}")
    print(f"Desviación estándar: {u_matrix.std():.3f}")

def print_cluster_info(results):
    """Imprime información de los clusters"""
    clusters = results['clusters']
    
    print("\n" + "="*50)
    print("INFORMACIÓN DE CLUSTERS")
    print("="*50)
    
    for cluster_key in sorted(clusters.keys()):
        countries = clusters[cluster_key]
        if len(countries) > 0:
            print(f"\nCluster {cluster_key}: {len(countries)} países")
            for country in countries:
                print(f"  • {country}")
    
    total_countries = sum(len(countries) for countries in clusters.values())
    active_clusters = sum(1 for countries in clusters.values() if len(countries) > 0)
    print(f"\n{'='*50}")
    print(f"Total: {total_countries} países en {active_clusters} clusters activos")
    print("="*50)

if __name__ == "__main__":
    print("Cargando resultados del clustering...")
    results = load_clustering_results()
    
    print_cluster_info(results)
    
    print("\nGenerando U-Matrix...")
    visualize_u_matrix(results)
