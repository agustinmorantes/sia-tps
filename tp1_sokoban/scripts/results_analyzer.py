import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any
import os


class ResultsAnalyzer:
    def __init__(self, results_file: str = "../results/performance_results.json"):
        self.results_file = results_file or "../results/performance_results.json"
        self.results_dir = "../results"
        
        # Crear carpeta results si no existe
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.data = self.load_results()
        
    def load_results(self) -> Dict:
        """Carga los resultados desde el archivo JSON"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"No se encontró el archivo de resultados: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def create_dataframe(self) -> pd.DataFrame:
        """Convierte los resultados a un DataFrame de pandas para análisis"""
        results = self.data["results"]
        
        # Crear DataFrame
        df = pd.DataFrame(results)
        
        # Agregar columnas derivadas
        df['map_name'] = df['map'].apply(lambda x: os.path.basename(x).replace('.txt', ''))
        df['algorithm_heuristic'] = df.apply(
            lambda row: f"{row['algorithm']}_{row['heuristic']}" if row['heuristic'] else row['algorithm'], 
            axis=1
        )
        
        return df
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Genera estadísticas resumidas de todos los resultados"""
        df = self.create_dataframe()
        
        summary = {
            "total_tests": len(df),
            "successful_tests": len(df[df['solution_found'] == True]),
            "failed_tests": len(df[df['solution_found'] == False]),
            "timeout_tests": len(df[df.get('timeout', False) == True]) if 'timeout' in df.columns else 0,
            "error_tests": len(df[df['error'].notna()]) if 'error' in df.columns else 0,
            
            "maps_tested": df['map_name'].unique().tolist(),
            "algorithms_tested": df['algorithm'].unique().tolist(),
            "heuristics_tested": df['heuristic'].dropna().unique().tolist(),
            
            "success_rate": len(df[df['solution_found'] == True]) / len(df) * 100,
            
            "avg_execution_time": df[df['solution_found'] == True]['execution_time'].mean(),
            "avg_nodes_expanded": df[df['solution_found'] == True]['nodes_expanded'].mean(),
            "avg_solution_length": df[df['solution_found'] == True]['solution_length'].mean(),
        }
        
        return summary
    
    def algorithm_comparison(self) -> pd.DataFrame:
        """Compara el rendimiento de diferentes algoritmos"""
        df = self.create_dataframe()
        
        # Agrupar por algoritmo y heurística
        comparison = df.groupby(['algorithm', 'heuristic']).agg({
            'solution_found': ['count', 'sum', 'mean'],
            'execution_time': ['mean', 'std', 'min', 'max'],
            'nodes_expanded': ['mean', 'std', 'min', 'max'],
            'solution_length': ['mean', 'std', 'min', 'max'],
            'border_nodes_count': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Renombrar columnas
        comparison.columns = [
            'total_tests', 'successful_tests', 'success_rate',
            'avg_time', 'std_time', 'min_time', 'max_time',
            'avg_nodes', 'std_nodes', 'min_nodes', 'max_nodes',
            'avg_solution_len', 'std_solution_len', 'min_solution_len', 'max_solution_len',
            'avg_border', 'std_border', 'min_border', 'max_border'
        ]
        
        return comparison
    
    def map_comparison(self) -> pd.DataFrame:
        """Compara el rendimiento en diferentes mapas"""
        df = self.create_dataframe()
        
        # Agrupar por mapa
        comparison = df.groupby('map_name').agg({
            'solution_found': ['count', 'sum', 'mean'],
            'execution_time': ['mean', 'std', 'min', 'max'],
            'nodes_expanded': ['mean', 'std', 'min', 'max'],
            'solution_length': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Renombrar columnas
        comparison.columns = [
            'total_tests', 'successful_tests', 'success_rate',
            'avg_time', 'std_time', 'min_time', 'max_time',
            'avg_nodes', 'std_nodes', 'min_nodes', 'max_nodes',
            'avg_solution_len', 'std_solution_len', 'min_solution_len', 'max_solution_len'
        ]
        
        return comparison
    
    def best_performing_combinations(self, top_n: int = 5) -> pd.DataFrame:
        """Encuentra las mejores combinaciones algoritmo-heurística"""
        df = self.create_dataframe()
        
        # Solo considerar pruebas exitosas
        successful = df[df['solution_found'] == True]
        
        if len(successful) == 0:
            return pd.DataFrame()
        
        # Agrupar por algoritmo y heurística
        performance = successful.groupby(['algorithm', 'heuristic']).agg({
            'execution_time': 'mean',
            'nodes_expanded': 'mean',
            'solution_length': 'mean',
            'solution_found': 'count'
        }).round(4)
        
        performance.columns = ['avg_time', 'avg_nodes', 'avg_solution_len', 'success_count']
        
        # Ordenar por tiempo de ejecución (mejor primero)
        performance = performance.sort_values('avg_time')
        
        return performance.head(top_n)
    
    def worst_performing_combinations(self, top_n: int = 5) -> pd.DataFrame:
        """Encuentra las peores combinaciones algoritmo-heurística"""
        df = self.create_dataframe()
        
        # Solo considerar pruebas exitosas
        successful = df[df['solution_found'] == True]
        
        if len(successful) == 0:
            return pd.DataFrame()
        
        # Agrupar por algoritmo y heurística
        performance = successful.groupby(['algorithm', 'heuristic']).agg({
            'execution_time': 'mean',
            'nodes_expanded': 'mean',
            'solution_length': 'mean',
            'solution_found': 'count'
        }).round(4)
        
        performance.columns = ['avg_time', 'avg_nodes', 'avg_solution_len', 'success_count']
        
        # Ordenar por tiempo de ejecución (peor primero)
        performance = performance.sort_values('avg_time', ascending=False)
        
        return performance.head(top_n)
    

    
    def export_to_csv(self, prefix: str = "../results/analysis"):
        """Exporta los resultados a archivos CSV para análisis externo"""
        df = self.create_dataframe()
        
        # Exportar datos completos
        df.to_csv(f"{prefix}_complete_data.csv", index=False)
        
        print(f"Archivos CSV exportados con prefijo: {prefix}")


def main():
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "../results/performance_results.json"
    
    try:
        analyzer = ResultsAnalyzer(results_file)
        
        print("Generando análisis de resultados...")
        
        # Exportar a CSV
        analyzer.export_to_csv()
        
        # Mostrar resumen en consola
        summary = analyzer.generate_summary_stats()
        print(f"\nResumen: {summary['successful_tests']}/{summary['total_tests']} pruebas exitosas ({summary['success_rate']:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrate de haber ejecutado primero performance_analyzer.py")
    except Exception as e:
        print(f"Error durante el análisis: {e}")


if __name__ == "__main__":
    main()
