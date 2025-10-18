import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys

def load_clustering_results(filename=None):
    """Carga los resultados del clustering desde un archivo JSON"""
    if filename is None:
        # Intentar cargar el último resultado
        try:
            filename = 'clustering_results_latest.json'
            with open(filename, 'r') as f:
                return json.load(f), filename
        except FileNotFoundError:
            # Fallback al archivo sin timestamp
            filename = 'clustering_results.json'
            with open(filename, 'r') as f:
                return json.load(f), filename
    else:
        with open(filename, 'r') as f:
            return json.load(f), filename

def calculate_heatmap_values(clusters, weights):
    """
    Calcula valores para el heatmap basados en características de los países
    """
    grid_size = weights.shape[0]
    heatmap_matrix = np.zeros((grid_size, grid_size))
    
    # Usar GDP promedio como valor principal (característica 1 en datos estandarizados)
    for i in range(grid_size):
        for j in range(grid_size):
            # El valor del heatmap será el GDP promedio de esa neurona
            # Como los datos están estandarizados, usamos el peso de la neurona
            if len(weights[i, j]) > 1:
                heatmap_matrix[i, j] = weights[i, j, 1]  # GDP es la característica 1
            else:
                heatmap_matrix[i, j] = weights[i, j, 0]  # Fallback a primera característica
    
    return heatmap_matrix

def visualize_kohonen_heatmap(results, filename, output_file='kohonen_heatmap.png'):
    """
    Visualiza el mapa de Kohonen como heatmap con colores que representan valores
    """
    clusters = results['clusters']
    grid_size = results['grid_size']
    
    # Obtener pesos si están disponibles
    weights_data = results.get('weights', None)
    if weights_data is not None:
        try:
            weights = np.array(weights_data)
            heatmap_values = calculate_heatmap_values(clusters, weights)
        except (ValueError, IndexError) as e:
            print(f"Error procesando pesos: {e}")
            print("Usando densidad de países como fallback...")
            # Fallback: usar densidad de países
            heatmap_values = np.zeros((grid_size, grid_size))
            for cluster_key, countries in clusters.items():
                i, j = eval(cluster_key)
                heatmap_values[i, j] = len(countries)
    else:
        # Fallback: usar densidad de países
        heatmap_values = np.zeros((grid_size, grid_size))
        for cluster_key, countries in clusters.items():
            i, j = eval(cluster_key)
            heatmap_values[i, j] = len(countries)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear heatmap
    im = ax.imshow(heatmap_values, cmap='viridis', interpolation='nearest')
    
    # Configurar ejes
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xlabel('', fontsize=12, fontweight='bold')
    ax.set_ylabel('', fontsize=12, fontweight='bold')
    
    # Título
    plt.title('Mapa de Kohonen - Heatmap de Países Europeos', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Añadir países en cada celda
    for i in range(grid_size):
        for j in range(grid_size):
            cluster_key = f"({i}, {j})"
            countries = clusters.get(cluster_key, [])
            
            if countries:
                # Determinar color del texto basado en el valor del heatmap
                text_color = 'white' if heatmap_values[i, j] < heatmap_values.mean() else 'black'
                
                # Crear texto con países
                country_text = '\n'.join(countries)
                ax.text(j, i, country_text,
                       ha='center', va='center',
                       fontsize=8 if len(countries) <= 4 else 7,
                       color=text_color,
                       fontweight='bold')
            else:
                ax.text(j, i, '(vacío)',
                       ha='center', va='center',
                       fontsize=8, style='italic',
                       color='white')
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('', fontsize=12, fontweight='bold')
    
    # Sin información adicional
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap guardado en: {output_file}")
    plt.show()

def print_cluster_statistics(results):
    """Imprime estadísticas de los clusters"""
    clusters = results['clusters']
    
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE CLUSTERING")
    print("="*60)
    
    for cluster_key in sorted(clusters.keys()):
        countries = clusters[cluster_key]
        if len(countries) > 0:
            print(f"\nCluster {cluster_key}: {len(countries)} países")
            for country in countries:
                print(f"  • {country}")
    
    total_countries = sum(len(countries) for countries in clusters.values())
    active_clusters = sum(1 for countries in clusters.values() if len(countries) > 0)
    print(f"\n{'='*60}")
    print(f"Total: {total_countries} países en {active_clusters} clusters activos")
    print("="*60)

if __name__ == "__main__":
    # Permitir especificar archivo como argumento
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("Cargando resultados del clustering...")
    results, loaded_file = load_clustering_results(filename)
    print(f"✓ Cargado desde: {loaded_file}")
    
    print_cluster_statistics(results)
    
    print("\nGenerando heatmap del mapa de Kohonen...")
    visualize_kohonen_heatmap(results, loaded_file)

