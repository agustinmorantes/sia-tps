# Trabajos Prácticos - Sistemas de Inteligencia Artificial

## Prerequisitos
- python
- pipenv

### Instalación pipenv
```shell
pip install --user pipenv
```

## Cómo correr los proyectos
### 1. Entrar a la carpeta del tp
### 2. Instalar dependencias
  ```shell
  pipenv install
  ```
### 3. Activar el virtual environment
  ```
  pipenv shell
  ```

## TP1 - Solucionador de Sokoban

### Ejecutar el juego básico
```shell
# Ejecutar con mapa por defecto (Map5.txt)
python main.py

# Ejecutar con un mapa específico
python main.py maps/Map3.txt
```

### Algoritmos disponibles
El solucionador implementa los siguientes algoritmos de búsqueda:
- **BFS** (Breadth-First Search)
- **DFS** (Depth-First Search)
- **Greedy**
- **A*** 

### Heurísticas implementadas
- **Manhattan** - Distancia ortogonal mínima
- **Manhattan Improved** - Manhattan + detección de deadlocks
- **Euclidean** - Distancia euclidiana
- **Chebyshev** - Distancia máxima entre coordenadas
- **Hamming** - Cuenta cajas fuera de lugar

### Ejecutar análisis de rendimiento completo
```shell
# Entrar al directorio de scripts
cd scripts

# Ejecutar análisis completo (puede tomar varios minutos)
python run_full_analysis.py

# O ejecutar componentes individuales:
python performance_analyzer.py  # Ejecuta las pruebas
python results_analyzer.py      # Analiza resultados
python plot_results.py          # Genera gráficos de métricas (nodos, costo, frontera)
python individual_map_graphs.py # Genera gráficos de duración por mapa
```

### Visualizar resultados
Los gráficos se generan automáticamente en la carpeta `graphs/`:

**Gráficos de métricas (`plot_results.py`):**
- `plot_Map3_nodes_expanded.png` - Nodos expandidos por algoritmo
- `plot_Map3_solution_cost.png` - Costo de solución por algoritmo
- `plot_Map3_border_nodes_count.png` - Nodos en frontera por algoritmo
- `plot_Map5_nodes_expanded.png` - Nodos expandidos por algoritmo
- `plot_Map5_solution_cost.png` - Costo de solución por algoritmo
- `plot_Map5_border_nodes_count.png` - Nodos en frontera por algoritmo

**Gráficos de duración (`individual_map_graphs.py`):**
- `plot_Map3_duration.png` - Tiempo de ejecución por algoritmo en Map3
- `plot_Map5_duration.png` - Tiempo de ejecución por algoritmo en Map5
- Gráficos de barras con errores estándar
- Colores específicos para cada algoritmo

### Configuración
Editar `config.json` para personalizar:
- Número de iteraciones por algoritmo
- Mapas a probar
- Timeout por prueba
- Algoritmos y heurísticas a evaluar

### Estructura del proyecto
```
tp1_sokoban/
├── main.py                 # Punto de entrada principal
├── config.json            # Configuración de pruebas
├── maps/                  # Mapas de Sokoban
├── src/                   # Código fuente
│   ├── classes/          # Clases principales
│   ├── map_parser.py     # Parser de mapas
│   └── map_viewer.py     # Visualización gráfica
├── scripts/              # Scripts de análisis
├── results/              # Resultados de pruebas
└── graphs/               # Gráficos generados
```

### Ejemplos de uso
```shell
# Jugar con visualización gráfica
python main.py maps/Map3.txt

# Ejecutar análisis rápido
cd scripts
python performance_analyzer.py

# Generar gráficos de métricas (nodos, costo, frontera)
python plot_results.py

# Generar gráficos de duración por mapa
python individual_map_graphs.py

# Ver resultados en CSV
cat results/analysis_complete_data.csv
```