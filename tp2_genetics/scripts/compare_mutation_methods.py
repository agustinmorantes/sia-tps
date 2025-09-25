#!/usr/bin/env python3
"""
Script para comparar métodos de mutación (single vs multi) mediante gráficos
de fitness vs generaciones y diversidad vs generaciones.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def load_metrics(config_name, outputs_dir):
    """Carga las métricas de un archivo metrics.json"""
    metrics_path = os.path.join(outputs_dir, config_name, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Warning: No se encontró {metrics_path}")
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error cargando {metrics_path}: {e}")
        return None

def extract_generation_data(metrics_data):
    """Extrae datos de generaciones del archivo de métricas"""
    if not metrics_data or 'gen_metrics' not in metrics_data:
        return None, None, None, None
    
    gen_metrics = metrics_data['gen_metrics']
    
    generations = [g['generation'] for g in gen_metrics]
    max_fitness = [g['max_fitness'] for g in gen_metrics]
    avg_fitness = [g['average_fitness'] for g in gen_metrics]
    diversity = [g['diversity'] for g in gen_metrics]
    
    return generations, max_fitness, avg_fitness, diversity

def get_config_pairs(outputs_dir):
    """Identifica pares de configuraciones (single vs multi)"""
    configs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    pairs = []
    
    # Buscar configuraciones que terminen en '_single'
    single_configs = [c for c in configs if c.endswith('_single')]
    
    for single_config in single_configs:
        # Buscar el par multi correspondiente
        base_name = single_config.replace('_single', '')
        multi_config = base_name + '_multi'
        
        if multi_config in configs:
            pairs.append((single_config, multi_config))
        else:
            # Intentar con correcciones ortográficas comunes
            corrected_base = base_name.replace('bolztmann', 'boltzmann')  # Corregir error ortográfico
            corrected_multi = corrected_base + '_multi'
            if corrected_multi in configs:
                pairs.append((single_config, corrected_multi))
                print(f"Info: Corregido nombre para {single_config} -> {corrected_multi}")
            else:
                print(f"Warning: No se encontró par para {single_config}")
    
    # También buscar configuraciones que terminen en '_multi' y no tengan par single
    multi_configs = [c for c in configs if c.endswith('_multi')]
    for multi_config in multi_configs:
        base_name = multi_config.replace('_multi', '')
        single_config = base_name + '_single'
        
        # Solo agregar si no está ya en la lista
        if single_config in configs and (single_config, multi_config) not in pairs:
            pairs.append((single_config, multi_config))
        else:
            # Intentar con correcciones ortográficas
            corrected_base = base_name.replace('boltzmann', 'bolztmann')  # Corregir error ortográfico
            corrected_single = corrected_base + '_single'
            if corrected_single in configs and (corrected_single, multi_config) not in pairs:
                pairs.append((corrected_single, multi_config))
                print(f"Info: Corregido nombre para {multi_config} -> {corrected_single}")
    
    return pairs

def create_comparison_plots(single_config, multi_config, outputs_dir, save_dir):
    """Crea gráficos de comparación para un par de configuraciones"""
    
    # Cargar datos
    single_data = load_metrics(single_config, outputs_dir)
    multi_data = load_metrics(multi_config, outputs_dir)
    
    if not single_data or not multi_data:
        print(f"Error: No se pudieron cargar datos para {single_config} o {multi_config}")
        return
    
    # Extraer datos de generaciones
    single_gens, single_max_fit, single_avg_fit, single_div = extract_generation_data(single_data)
    multi_gens, multi_max_fit, multi_avg_fit, multi_div = extract_generation_data(multi_data)
    
    if not all([single_gens, multi_gens]):
        print(f"Error: No se pudieron extraer datos de generaciones para {single_config} o {multi_config}")
        return
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Configurar el nombre base para los títulos
    base_name = single_config.replace('_single', '').replace('_', ' ').title()
    
    # Gráfico 1: Fitness vs Generaciones (Max y Average)
    ax1.plot(single_gens, single_max_fit, 'b-', label='Single', linewidth=2)
    ax1.plot(multi_gens, multi_max_fit, 'r-', label='Multi', linewidth=2)
    
    ax1.set_xlabel('Generaciones')
    ax1.set_ylabel('Fitness')
    ax1.set_title(f'Fitness vs Generaciones - {base_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Gráfico 2: Diversidad vs Generaciones
    ax2.plot(single_gens, single_div, 'b-', label='Single', linewidth=2)
    ax2.plot(multi_gens, multi_div, 'r-', label='Multi', linewidth=2)
    
    ax2.set_xlabel('Generaciones')
    ax2.set_ylabel('Diversidad')
    ax2.set_title(f'Diversidad vs Generaciones - {base_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    filename = f"comparison_{base_name.replace(' ', '_').lower()}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico combinado guardado: {save_path}")
    
    # Crear gráfico separado solo de fitness para mejor visualización
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(single_gens, single_max_fit, 'b-', label='Single', linewidth=2)
    ax.plot(multi_gens, multi_max_fit, 'r-', label='Multi', linewidth=2)
    
    ax.set_xlabel('Generaciones')
    ax.set_ylabel('Fitness')
    ax.set_title(f'Fitness vs Generaciones - {base_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    fitness_filename = f"fitness_{base_name.replace(' ', '_').lower()}.png"
    fitness_save_path = os.path.join(save_dir, fitness_filename)
    plt.savefig(fitness_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de fitness guardado: {fitness_save_path}")
    
    # Crear gráfico separado solo de diversidad
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(single_gens, single_div, 'b-', label='Single', linewidth=2)
    ax.plot(multi_gens, multi_div, 'r-', label='Multi', linewidth=2)
    
    ax.set_xlabel('Generaciones')
    ax.set_ylabel('Diversidad')
    ax.set_title(f'Diversidad vs Generaciones - {base_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    diversity_filename = f"diversity_{base_name.replace(' ', '_').lower()}.png"
    diversity_save_path = os.path.join(save_dir, diversity_filename)
    plt.savefig(diversity_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de diversidad guardado: {diversity_save_path}")
    
    # Crear gráfico de convergencia (diferencia entre max y avg fitness)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calcular convergencia (diferencia entre max y avg fitness)
    single_convergence = [max_f - avg_f for max_f, avg_f in zip(single_max_fit, single_avg_fit)]
    multi_convergence = [max_f - avg_f for max_f, avg_f in zip(multi_max_fit, multi_avg_fit)]
    
    ax.plot(single_gens, single_convergence, 'b-', label='Single', linewidth=2)
    ax.plot(multi_gens, multi_convergence, 'r-', label='Multi', linewidth=2)
    
    ax.set_xlabel('Generaciones')
    ax.set_ylabel('Convergencia (Max - Avg Fitness)')
    ax.set_title(f'Convergencia vs Generaciones - {base_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    convergence_filename = f"convergence_{base_name.replace(' ', '_').lower()}.png"
    convergence_save_path = os.path.join(save_dir, convergence_filename)
    plt.savefig(convergence_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de convergencia guardado: {convergence_save_path}")

def create_summary_table(pairs, outputs_dir, save_dir):
    """Crea una tabla resumen con métricas finales de todos los pares"""
    
    summary_data = []
    
    for single_config, multi_config in pairs:
        single_data = load_metrics(single_config, outputs_dir)
        multi_data = load_metrics(multi_config, outputs_dir)
        
        if single_data and multi_data:
            summary_data.append({
                'Config': single_config.replace('_single', ''),
                'Single_Final_Fitness': single_data.get('fitness', 0),
                'Single_Generations': single_data.get('generations', 0),
                'Single_Runtime': single_data.get('runtime_seconds', 0),
                'Multi_Final_Fitness': multi_data.get('fitness', 0),
                'Multi_Generations': multi_data.get('generations', 0),
                'Multi_Runtime': multi_data.get('runtime_seconds', 0),
                'Fitness_Diff': multi_data.get('fitness', 0) - single_data.get('fitness', 0),
                'Runtime_Diff': multi_data.get('runtime_seconds', 0) - single_data.get('runtime_seconds', 0)
            })
    
    # Crear tabla usando matplotlib
    if summary_data:
        fig, ax = plt.subplots(figsize=(16, len(summary_data) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Preparar datos para la tabla
        headers = ['Config', 'Single Fitness', 'Multi Fitness', 'Fitness Diff', 
                  'Single Gen', 'Multi Gen', 'Single Time (s)', 'Multi Time (s)', 'Time Diff (s)']
        
        table_data = []
        for row in summary_data:
            table_data.append([
                row['Config'],
                f"{row['Single_Final_Fitness']:.4f}",
                f"{row['Multi_Final_Fitness']:.4f}",
                f"{row['Fitness_Diff']:+.4f}",
                f"{row['Single_Generations']}",
                f"{row['Multi_Generations']}",
                f"{row['Single_Runtime']:.1f}",
                f"{row['Multi_Runtime']:.1f}",
                f"{row['Runtime_Diff']:+.1f}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Colorear diferencias
        for i in range(1, len(table_data) + 1):
            # Fitness difference
            fitness_diff = float(table_data[i-1][3])
            if fitness_diff > 0:
                table[(i, 3)].set_facecolor('#90EE90')  # Verde claro
            elif fitness_diff < 0:
                table[(i, 3)].set_facecolor('#FFB6C1')  # Rosa claro
            
            # Time difference
            time_diff = float(table_data[i-1][8])
            if time_diff > 0:
                table[(i, 8)].set_facecolor('#FFB6C1')  # Rosa claro (más tiempo = peor)
            elif time_diff < 0:
                table[(i, 8)].set_facecolor('#90EE90')  # Verde claro (menos tiempo = mejor)
        
        plt.title('Resumen Comparativo: Single vs Multi Mutation', fontsize=14, fontweight='bold', pad=20)
        
        summary_path = os.path.join(save_dir, 'comparison_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tabla resumen guardada: {summary_path}")

def main():
    """Función principal"""
    # Configurar directorios
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs"
    save_dir = base_dir / "graphs" / "mutation_comparison"
    
    # Crear directorio de salida si no existe
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Buscando configuraciones en: {outputs_dir}")
    print(f"Guardando gráficos en: {save_dir}")
    
    # Verificar que el directorio de outputs existe
    if not outputs_dir.exists():
        print(f"Error: No se encontró el directorio {outputs_dir}")
        return
    
    # Obtener pares de configuraciones
    pairs = get_config_pairs(outputs_dir)
    print(f"\nEncontrados {len(pairs)} pares de configuraciones:")
    for i, (single, multi) in enumerate(pairs, 1):
        print(f"  {i:2d}. {single} vs {multi}")
    
    if len(pairs) == 0:
        print("No se encontraron pares de configuraciones. Verificando directorio...")
        configs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
        print(f"Configuraciones disponibles: {configs}")
        return
    
    # Crear gráficos para cada par
    successful_plots = 0
    failed_plots = 0
    
    for i, (single_config, multi_config) in enumerate(pairs, 1):
        print(f"\n{'='*60}")
        print(f"Procesando par {i}/{len(pairs)}: {single_config} vs {multi_config}")
        print(f"{'='*60}")
        
        try:
            create_comparison_plots(single_config, multi_config, outputs_dir, save_dir)
            successful_plots += 1
            print(f"✅ Par {i} procesado exitosamente")
        except Exception as e:
            failed_plots += 1
            print(f"❌ Error procesando par {i}: {e}")
            continue
    
    # Crear tabla resumen
    print(f"\n{'='*60}")
    print("Creando tabla resumen...")
    print(f"{'='*60}")
    
    try:
        create_summary_table(pairs, outputs_dir, save_dir)
        print("✅ Tabla resumen creada exitosamente")
    except Exception as e:
        print(f"❌ Error creando tabla resumen: {e}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"Total de pares encontrados: {len(pairs)}")
    print(f"Gráficos creados exitosamente: {successful_plots}")
    print(f"Gráficos fallidos: {failed_plots}")
    print(f"Gráficos guardados en: {save_dir}")
    
    if successful_plots > 0:
        print(f"\n🎉 ¡Análisis completado! Se generaron {successful_plots} conjuntos de gráficos.")
    else:
        print(f"\n⚠️  No se pudieron generar gráficos. Revisa los errores anteriores.")

if __name__ == "__main__":
    main()
