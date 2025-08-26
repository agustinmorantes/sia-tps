import json
import os
import sys
import time
import signal
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

from src.classes.heuristics.ManhattanImprovedHeuristic import ManhattanImprovedHeuristic

# Agregar el directorio padre al path para poder importar src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar las clases del proyecto
from src.classes.Point import Point
from src.classes.Sokoban import SokobanManager
from src.classes.heuristics.ManhattanHeuristic import ManhattanHeuristic
from src.classes.heuristics.EuclideanHeuristic import EuclideanHeuristic
from src.classes.heuristics.ChebyshevHeuristic import ChebyshevHeuristic
from src.classes.heuristics.HammingHeuristic import HammingHeuristic
from src.map_parser import load_and_parse_map


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


class PerformanceAnalyzer:
    def __init__(self, config_file: str = "../config.json"):
        self.config = self.load_config(config_file)
        self.results = []
        self.heuristics = {
            "Manhattan": ManhattanHeuristic(),
            "Manhattan Improved": ManhattanImprovedHeuristic(),
            "Euclidean": EuclideanHeuristic(),
            "Chebyshev": ChebyshevHeuristic(),
            "Hamming": HammingHeuristic()
        }
        
        # Crear carpeta results si no existe
        self.results_dir = "../results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
    def load_config(self, config_file: str) -> Dict:
        """Carga la configuración desde el archivo JSON"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo de configuración {config_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: El archivo de configuración no es un JSON válido: {e}")
            sys.exit(1)
    
    def load_map(self, map_path: str) -> tuple:
        """Carga y parsea un mapa"""
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Mapa no encontrado: {map_path}")
        
        map_data, walls, goals, boxes, player_pos, size = load_and_parse_map(map_path)
        
        player = Point(*player_pos)
        boxes_points = {Point(x, y) for x, y in boxes}
        walls_points = {Point(x, y) for x, y in walls}
        goals_points = {Point(x, y) for x, y in goals}
        
        return player, boxes_points, walls_points, goals_points, size
    
    def run_algorithm(self, sokoban: SokobanManager, algorithm_name: str, heuristic_name: Optional[str] = None) -> Dict[str, Any]:
        """Ejecuta un algoritmo específico y retorna las estadísticas"""
        sokoban.reset()
        
        # Configurar timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config["test_config"]["timeout_seconds"])
        
        try:
            start_time = time.time()
            
            if algorithm_name == "BFS":
                solution = sokoban.bfs()
            elif algorithm_name == "DFS":
                solution = sokoban.dfs()
            elif algorithm_name == "Greedy":
                if heuristic_name is None:
                    raise ValueError("Greedy requiere una heurística")
                heuristic = self.heuristics[heuristic_name]
                solution = sokoban.greedy(heuristic)
            elif algorithm_name == "A*":
                if heuristic_name is None:
                    raise ValueError("A* requiere una heurística")
                heuristic = self.heuristics[heuristic_name]
                solution = sokoban.a_star(heuristic)
            else:
                raise ValueError(f"Algoritmo no reconocido: {algorithm_name}")
            
            signal.alarm(0)  # Cancelar timeout
            
            stats = sokoban.get_statistics()
            stats["algorithm"] = algorithm_name
            stats["heuristic"] = heuristic_name
            stats["solution_found"] = solution is not None
            stats["solution_length"] = len(solution) - 1 if solution else 0
            stats["total_time"] = time.time() - start_time
            
            return stats
            
        except TimeoutError:
            signal.alarm(0)
            return {
                "algorithm": algorithm_name,
                "heuristic": heuristic_name,
                "solution_found": False,
                "solution_length": 0,
                "nodes_expanded": 0,
                "border_nodes_count": 0,
                "execution_time": self.config["test_config"]["timeout_seconds"],
                "total_time": self.config["test_config"]["timeout_seconds"],
                "timeout": True
            }
        except Exception as e:
            signal.alarm(0)
            return {
                "algorithm": algorithm_name,
                "heuristic": heuristic_name,
                "solution_found": False,
                "solution_length": 0,
                "nodes_expanded": 0,
                "border_nodes_count": 0,
                "execution_time": 0,
                "total_time": 0,
                "error": str(e)
            }
    
    def run_single_test(self, map_path: str, algorithm_config: Dict, iteration: int) -> Dict[str, Any]:
        """Ejecuta una sola prueba para un mapa, algoritmo e iteración específicos"""
        print(f"Ejecutando {algorithm_config['name']} con heurística {algorithm_config['heuristic']} en {map_path} - Iteración {iteration + 1}")
        
        try:
            # Cargar mapa
            player, boxes, walls, goals, size = self.load_map(map_path)
            
            # Crear instancia de Sokoban
            sokoban = SokobanManager(walls=walls, goals=goals, player=player, boxes=boxes, size=size)
            
            # Ejecutar algoritmo
            stats = self.run_algorithm(sokoban, algorithm_config["name"], algorithm_config["heuristic"])
            
            # Agregar información adicional
            result = {
                "timestamp": datetime.now().isoformat(),
                "map": map_path,
                "iteration": iteration + 1,
                **stats
            }
            
            return result
            
        except Exception as e:
            print(f"Error en prueba: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "map": map_path,
                "iteration": iteration + 1,
                "algorithm": algorithm_config["name"],
                "heuristic": algorithm_config["heuristic"],
                "solution_found": False,
                "solution_length": 0,
                "nodes_expanded": 0,
                "border_nodes_count": 0,
                "execution_time": 0,
                "total_time": 0,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas según la configuración"""
        print("Iniciando análisis de rendimiento...")
        print(f"Mapas: {len(self.config['test_config']['maps'])}")
        print(f"Algoritmos: {len(self.config['test_config']['algorithms'])}")
        print(f"Iteraciones por algoritmo: {self.config['test_config']['iterations_per_algorithm']}")
        print(f"Total de pruebas: {len(self.config['test_config']['maps']) * len(self.config['test_config']['algorithms']) * self.config['test_config']['iterations_per_algorithm']}")
        print("-" * 50)
        
        total_tests = 0
        completed_tests = 0
        
        for map_path in self.config["test_config"]["maps"]:
            if not os.path.exists(map_path):
                print(f"Advertencia: Mapa {map_path} no encontrado, saltando...")
                continue
                
            for algorithm_config in self.config["test_config"]["algorithms"]:
                for iteration in range(self.config["test_config"]["iterations_per_algorithm"]):
                    total_tests += 1
                    
                    result = self.run_single_test(map_path, algorithm_config, iteration)
                    self.results.append(result)
                    
                    completed_tests += 1
                    
                    # Mostrar progreso
                    if completed_tests % 10 == 0:
                        print(f"Progreso: {completed_tests}/{total_tests} pruebas completadas")
        
        print(f"\nAnálisis completado. {completed_tests} pruebas ejecutadas.")
    
    def save_results(self):
        """Guarda los resultados en el archivo especificado en la configuración"""
        output_file = os.path.join(self.results_dir, self.config["test_config"]["output_file"])
        
        # Crear estructura de resultados
        output_data = {
            "test_config": self.config["test_config"],
            "summary": {
                "total_tests": len(self.results),
                "timestamp": datetime.now().isoformat(),
                "maps_tested": list(set(r["map"] for r in self.results)),
                "algorithms_tested": list(set(r["algorithm"] for r in self.results))
            },
            "results": self.results
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Resultados guardados en: {output_file}")
        except Exception as e:
            print(f"Error guardando resultados: {e}")
    
    def print_summary(self):
        """Imprime un resumen de los resultados"""
        if not self.results:
            print("No hay resultados para mostrar")
            return
        
        print("\n" + "="*60)
        print("RESUMEN DE RESULTADOS")
        print("="*60)
        
        # Agrupar por mapa y algoritmo
        by_map_algorithm = {}
        for result in self.results:
            key = (result["map"], result["algorithm"], result["heuristic"])
            if key not in by_map_algorithm:
                by_map_algorithm[key] = []
            by_map_algorithm[key].append(result)
        
        for (map_name, algorithm, heuristic), results in by_map_algorithm.items():
            print(f"\n{map_name} - {algorithm} ({heuristic})")
            print("-" * 40)
            
            # Calcular estadísticas
            successful = [r for r in results if r["solution_found"]]
            timeouts = [r for r in results if r.get("timeout", False)]
            errors = [r for r in results if "error" in r]
            
            print(f"  Pruebas exitosas: {len(successful)}/{len(results)}")
            print(f"  Timeouts: {len(timeouts)}")
            print(f"  Errores: {len(errors)}")
            
            if successful:
                avg_time = sum(r["execution_time"] for r in successful) / len(successful)
                avg_nodes = sum(r["nodes_expanded"] for r in successful) / len(successful)
                avg_solution_length = sum(r["solution_length"] for r in successful) / len(successful)
                
                print(f"  Tiempo promedio: {avg_time:.4f}s")
                print(f"  Nodos expandidos promedio: {avg_nodes:.0f}")
                print(f"  Longitud de solución promedio: {avg_solution_length:.1f}")


def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "../config.json"
    
    analyzer = PerformanceAnalyzer(config_file)
    
    try:
        analyzer.run_all_tests()
        analyzer.save_results()
        analyzer.print_summary()
    except KeyboardInterrupt:
        print("\nAnálisis interrumpido por el usuario")
        analyzer.save_results()
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        traceback.print_exc()
        analyzer.save_results()


if __name__ == "__main__":
    main()
