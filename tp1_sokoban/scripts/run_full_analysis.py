#!/usr/bin/env python3
"""
Script para ejecutar el análisis completo de rendimiento del solver de Sokoban.
Este script ejecuta todas las pruebas con la configuración principal y genera
todos los reportes y archivos CSV para análisis posterior.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def print_header(title):
    """Imprime un encabezado formateado"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def print_step(step, description):
    """Imprime información de un paso"""
    print(f"\n📋 PASO {step}: {description}")
    print("-" * 60)

def run_command(command, description, use_pipenv=True):
    """Ejecuta un comando y muestra el progreso"""
    print(f"Ejecutando: {description}")
    print(f"Comando: {command}")
    
    start_time = time.time()
    
    try:
        if use_pipenv:
            result = subprocess.run(f"pipenv run {command}", shell=True, check=True, capture_output=False)
        else:
            result = subprocess.run(command, shell=True, check=True, capture_output=False)
        
        end_time = time.time()
        print(f"✅ Completado en {end_time - start_time:.2f} segundos")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ Interrumpido por el usuario")
        return False

def main():
    print_header("ANÁLISIS COMPLETO DE RENDIMIENTO - SOKOBAN SOLVER")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("../config.json"):
        print("❌ Error: No se encontró config.json")
        print("Asegúrate de estar en el directorio scripts")
        sys.exit(1)
    
    # Verificar que pandas esté instalado
    print_step(1, "Verificando dependencias")
    try:
        import pandas
        print("✅ pandas está instalado")
    except ImportError:
        print("❌ pandas no está instalado. Instalando...")
        if not run_command("pipenv install pandas", "Instalando pandas", use_pipenv=False):
            print("❌ No se pudo instalar pandas")
            sys.exit(1)
    
    # Paso 2: Ejecutar pruebas de rendimiento
    print_step(2, "Ejecutando pruebas de rendimiento")
    print("Este paso puede tomar varios minutos dependiendo de la configuración...")
    
    if not run_command("python performance_analyzer.py", "Ejecutando análisis de rendimiento"):
        print("❌ Falló la ejecución de las pruebas")
        sys.exit(1)
    
    # Paso 3: Verificar que se generaron los resultados
    results_file = "../results/performance_results.json"
    if not os.path.exists(results_file):
        print("❌ No se encontró performance_results.json")
        print("Las pruebas pueden haber fallado")
        sys.exit(1)
    
    # Paso 4: Generar análisis de resultados
    print_step(3, "Generando análisis de resultados")
    
    if not run_command("python results_analyzer.py", "Generando análisis detallado"):
        print("❌ Falló el análisis de resultados")
        sys.exit(1)
    
    # Paso 5: Mostrar resumen final
    print_step(4, "Resumen final")
    
    print("\n📊 ARCHIVOS GENERADOS:")
    
    files_generated = []
    expected_files = [
        "../results/performance_results.json",
        "../results/analysis_complete_data.csv"
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
            files_generated.append(file)
        else:
            print(f"❌ {file} (no encontrado)")
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"- Archivos generados: {len(files_generated)}/{len(expected_files)}")
    
    print_header("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Análisis interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)
