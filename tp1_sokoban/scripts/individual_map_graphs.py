#!/usr/bin/env python3
"""
Script para generar un gráfico individual para cada mapa.
Crea un gráfico separado para cada mapa mostrando la duración de todos los algoritmos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def create_map_graph(df, map_name, output_dir="../graphs/"):
    """Crea un gráfico individual para un mapa específico"""
    
    # Filtrar datos para el mapa específico
    map_data = df[df['map_name'] == map_name]
    
    # Calcular duración promedio y desviación estándar por algoritmo (en milisegundos)
    duration_data = map_data.groupby('algorithm_heuristic')['execution_time'].agg(['mean', 'std']).reset_index()
    duration_data['mean'] *= 1000  # Convertir a milisegundos
    duration_data['std'] *= 1000   # Convertir a milisegundos
    duration_data = duration_data.sort_values('mean')
    
    # Definir colores específicos para cada algoritmo
    colors = {
        'BFS': '#FF6B6B',           # Rojo
        'DFS': '#4ECDC4',           # Turquesa
        'Greedy_Manhattan': '#45B7D1',   # Azul claro
        'Greedy_Euclidean': '#96CEB4',   # Verde claro
        'Greedy_Chebyshev': '#FFEAA7',   # Amarillo
        'Greedy_Hamming': '#DDA0DD',     # Lavanda
        'A*_Manhattan': '#FF8C42',       # Naranja
        'A*_Euclidean': '#6A5ACD',       # Slate blue
        'A*_Chebyshev': '#20B2AA',       # Light sea green
        'A*_Hamming': '#FF69B4'          # Hot pink
    }
    
    # Configurar el gráfico
    plt.figure(figsize=(12, 8))
    
    # Crear gráfico de barras con barras de error (desviación estándar)
    bars = plt.bar(
        range(len(duration_data)),
        duration_data['mean'],
        yerr=duration_data['std'],
        capsize=6
    )
    
    # Aplicar colores específicos a cada barra
    for i, (_, row) in enumerate(duration_data.iterrows()):
        algorithm = row['algorithm_heuristic']
        bars[i].set_color(colors.get(algorithm, '#CCCCCC'))  # Gris por defecto si no encuentra el color
    
    # Personalizar el gráfico
    plt.title(f'Duración Promedio por Algoritmo - {map_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (milisegundos)', fontsize=12)

    # Configurar etiquetas del eje X
    plt.xticks(range(len(duration_data)), duration_data['algorithm_heuristic'], rotation=45, ha='right')
    
    # Agregar valores en las barras
    for i, (_, row) in enumerate(duration_data.iterrows()):
        plt.text(i, row['mean'] + max(duration_data['mean']) * 0.01, 
                f'{row["mean"]:.2f}', ha='center', va='bottom', fontsize=10)

    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    output_file = os.path.join(output_dir, f"plot_{map_name}_duration.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_file}")
    
    # Cerrar la figura para liberar memoria
    plt.close()
    
    return duration_data

def main():
    """Función principal"""
    print("📊 Generando gráficos individuales por mapa...")
    
    # Cargar datos
    results_file = "../results/performance_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Error: No se encontró {results_file}")
        print("Asegúrate de haber ejecutado primero el análisis de rendimiento")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convertir a DataFrame
    df = pd.DataFrame(data["results"])
    df['map_name'] = df['map'].apply(lambda x: os.path.basename(x).replace('.txt', ''))
    df['algorithm_heuristic'] = df.apply(
        lambda row: f"{row['algorithm']}_{row['heuristic']}" if row['heuristic'] else row['algorithm'], 
        axis=1
    )
    
    # Obtener lista de mapas únicos
    maps = df['map_name'].unique()
    print(f"📋 Mapas encontrados: {maps}")
    
    # Crear gráfico para cada mapa
    all_results = {}
    for map_name in maps:
        print(f"\n🎨 Generando gráfico para {map_name}...")
        results = create_map_graph(df, map_name)
        all_results[map_name] = results
    
    # Mostrar resumen
    print(f"\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"📁 Gráficos generados: {len(maps)}")
    
    # Mostrar archivos generados
    print(f"\n📄 Archivos generados:")
    for map_name in maps:
        filename = f"plot_{map_name}_duration.png"
        print(f"   - {filename}")

if __name__ == "__main__":
    main()
