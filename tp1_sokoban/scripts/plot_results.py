import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def format_algorithm_name(algorithm: str, heuristic: str = None) -> str:
    """Formatea el nombre del algoritmo para mostrar"""
    if algorithm == "A*":
        return f"A* ({heuristic})" if heuristic else "A*"
    elif algorithm == "Greedy":
        return f"Greedy ({heuristic})" if heuristic else "Greedy"
    else:
        return algorithm

def plot_results():
    results_dir = "../results"
    output_dir = "../graphs"
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos desde performance_results.json
    results_file = os.path.join(results_dir, "performance_results.json")
    
    if not os.path.exists(results_file):
        print(f"Error: No se encontró {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convertir a DataFrame para facilitar el análisis
    df = pd.DataFrame(data['results'])
    
    # Filtrar solo resultados exitosos
    df_success = df[df['solution_found'] == True].copy()
    
    if df_success.empty:
        print("No hay datos de soluciones exitosas para graficar")
        return
    
    # Crear nombre formateado del algoritmo
    df_success['algorithm_formatted'] = df_success.apply(
        lambda row: format_algorithm_name(row['algorithm'], row['heuristic']), 
        axis=1
    )
    
    # Obtener mapas únicos
    maps = df_success['map'].unique()
    
    # Orden de algoritmos para consistencia
    algo_order = [
        "BFS", "DFS",
        "Greedy (Manhattan)", "Greedy (Manhattan Improved)", "Greedy (Euclidean)", "Greedy (Chebyshev)", "Greedy (Hamming)",
        "A* (Manhattan)", "A* (Manhattan Improved)", "A* (Euclidean)", "A* (Chebyshev)", "A* (Hamming)"
    ]
    
    # Métricas a graficar
    metrics = {
        "nodes_expanded": "Nodos Expandidos",
        "solution_cost": "Costo de Solución (Movimientos)",
        "border_nodes_count": "Máximo Nodos Frontera"
    }
    
    for map_name in maps:
        # Extraer nombre del mapa sin la ruta
        map_short_name = os.path.basename(map_name).replace('.txt', '')
        
        # Filtrar datos para este mapa
        map_data = df_success[df_success['map'] == map_name]
        
        if map_data.empty:
            print(f"No hay datos para {map_short_name}")
            continue
        
        # Calcular promedios por algoritmo
        map_stats = map_data.groupby('algorithm_formatted').agg({
            'nodes_expanded': 'mean',
            'solution_cost': 'mean', 
            'border_nodes_count': 'mean'
        }).round(2)
        
        # Filtrar algoritmos que están en el orden deseado
        available_algorithms = [algo for algo in algo_order if algo in map_stats.index]
        
        if not available_algorithms:
            print(f"No hay algoritmos válidos para {map_short_name}")
            continue
        
        # Generar gráficos para cada métrica
        for metric_key, metric_title in metrics.items():
            if metric_key not in map_stats.columns:
                continue
                
            values = [map_stats.loc[algo, metric_key] for algo in available_algorithms]
            
            # Colores específicos para cada algoritmo
            colors = {
                'BFS': '#FF6B6B',           # Rojo
                'DFS': '#4ECDC4',           # Turquesa
                'Greedy (Manhattan)': '#45B7D1',   # Azul claro
                'Greedy (Euclidean)': '#96CEB4',   # Verde claro
                'Greedy (Chebyshev)': '#FFEAA7',   # Amarillo
                'Greedy (Hamming)': '#DDA0DD',     # Lavanda
                'A* (Manhattan)': '#FF8C42',       # Naranja
                'A* (Euclidean)': '#6A5ACD',       # Slate blue
                'A* (Chebyshev)': '#20B2AA',       # Light sea green
                'A* (Hamming)': '#FF69B4'          # Hot pink
            }
            
            # Crear figura
            plt.figure(figsize=(12, 8))
            
            # Crear barras con colores específicos
            bar_colors = [colors.get(algo, '#CCCCCC') for algo in available_algorithms]
            bars = plt.barh(available_algorithms, values, color=bar_colors, edgecolor='black', alpha=0.8)
            
            # Configurar ejes
            max_val = max(values) if values else 1
            plt.xlim(0, max_val * 1.1)
            
            # Agregar valores en las barras
            for bar, value in zip(bars, values):
                xval = bar.get_width()
                intValue = int(value)
                plt.text(xval + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                        f"{intValue}".replace(",", "."),
                        va='center', ha='left', fontsize=10, fontweight='bold')
            
            # Configurar título y etiquetas
            plt.ylabel("Algoritmo", fontsize=12, fontweight='bold')
            plt.xlabel(metric_title, fontsize=12, fontweight='bold')
            plt.title(f"{metric_title} por Algoritmo - {map_short_name}", 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Agregar grid
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Ajustar layout y guardar
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"plot_{map_short_name}_{metric_key}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Gráfico generado: {output_path}")
    
    print(f"\n🎉 Todos los gráficos han sido guardados en: {output_dir}")

if __name__ == "__main__":
    plot_results()
