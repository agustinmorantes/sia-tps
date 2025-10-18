import json
import matplotlib.pyplot as plt
import numpy as np

def load_clustering_results(filename='clustering_results_latest.json'):
    """Carga los resultados del clustering desde un archivo JSON"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_feature_heatmap(weights, feature_index, feature_name, output_file=None):
    """
    Crea un heatmap para una característica específica
    """
    grid_size = weights.shape[0]
    
    # Extraer valores de la característica específica
    feature_values = weights[:, :, feature_index]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Crear heatmap
    im = ax.imshow(feature_values, cmap='viridis', interpolation='nearest')
    
    # Configurar ejes
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    
    # Título
    plt.title(f'{feature_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Añadir valores en cada celda
    for i in range(grid_size):
        for j in range(grid_size):
            # Determinar color del texto basado en el valor de la celda
            cell_value = feature_values[i, j]
            min_val = feature_values.min()
            max_val = feature_values.max()
            
            # Normalizar el valor entre 0 y 1
            normalized_value = (cell_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            # Usar texto blanco para valores bajos (fondos oscuros) y negro para valores altos (fondos claros)
            text_color = 'white' if normalized_value < 0.5 else 'black'
            
            ax.text(j, i, f'{feature_values[i, j]:.2f}',
                   ha="center", va="center", 
                   color=text_color, fontsize=12, fontweight='bold')
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap de {feature_name} guardado en: {output_file}")
    
    plt.show()
    
    return feature_values

def create_multiple_feature_heatmaps(weights, feature_indices, feature_names, output_file=None):
    """
    Crea múltiples heatmaps en una sola figura
    """
    n_features = len(feature_indices)
    fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 6))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, (feature_idx, feature_name) in enumerate(zip(feature_indices, feature_names)):
        # Extraer valores de la característica específica
        feature_values = weights[:, :, feature_idx]
        
        # Crear heatmap
        im = axes[idx].imshow(feature_values, cmap='viridis', interpolation='nearest')
        
        # Configurar ejes
        axes[idx].set_xticks(range(weights.shape[0]))
        axes[idx].set_yticks(range(weights.shape[0]))
        axes[idx].set_title(f'{feature_name}', fontsize=14, fontweight='bold')
        
        # Añadir valores en cada celda
        for i in range(weights.shape[0]):
            for j in range(weights.shape[0]):
                # Determinar color del texto
                cell_value = feature_values[i, j]
                min_val = feature_values.min()
                max_val = feature_values.max()
                
                normalized_value = (cell_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                text_color = 'white' if normalized_value < 0.5 else 'black'
                
                axes[idx].text(j, i, f'{feature_values[i, j]:.2f}',
                             ha="center", va="center", 
                             color=text_color, fontsize=10, fontweight='bold')
        
        # Añadir colorbar
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmaps guardados en: {output_file}")
    
    plt.show()

def main():
    # Cargar resultados
    print("Cargando resultados del clustering...")
    results = load_clustering_results()
    
    # Obtener pesos
    weights_data = results.get('weights', None)
    if weights_data is None:
        print("Error: No se encontraron pesos en los resultados.")
        print("Ejecuta main.py primero para generar los pesos.")
        return
    
    weights = np.array(weights_data)
    
    # Definir características disponibles
    features = {
        'Area': 0,
        'GDP': 1,
        'Inflation': 2,
        'Life.expect': 3,
        'Military': 4,
        'Pop.growth': 5,
        'Unemployment': 6
    }
    
    print("\nCaracterísticas disponibles:")
    for name, idx in features.items():
        print(f"  {idx}: {name}")
    
    # Crear heatmaps para Life.expect y Pop.growth
    print("\nGenerando heatmaps para Life.expect y Pop.growth...")
    create_multiple_feature_heatmaps(
        weights, 
        [features['Life.expect'], features['Pop.growth']], 
        ['Life.expect', 'Pop.growth'],
        'life_expect_pop_growth_heatmaps.png'
    )
    
    # Opción para crear heatmap individual
    print("\n¿Quieres crear un heatmap individual? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        print("Ingresa el nombre de la característica:")
        feature_name = input()
        
        if feature_name in features:
            feature_idx = features[feature_name]
            print(f"Generando heatmap para {feature_name}...")
            create_feature_heatmap(weights, feature_idx, feature_name, f'{feature_name.lower()}_heatmap.png')
        else:
            print(f"Característica '{feature_name}' no encontrada.")

if __name__ == "__main__":
    main()
