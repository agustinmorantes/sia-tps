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

## TP2 - Algoritmo Genético para Aproximación de Imágenes

### Ejecutar el algoritmo genético
Uso: 
```shell
python main.py [archivo de configuración]
```

Ejemplo:
```shell
python main.py config.json
```

### Algoritmos de selección disponibles
El algoritmo genético implementa los siguientes métodos de selección:
- **Elite** - Selección elitista
- **Roulette** - Selección por ruleta
- **Ranking** - Selección por ranking
- **Tournament Deterministic** - Torneo determinístico
- **Tournament Probabilistic** - Torneo probabilístico
- **Universal** - Selección universal estocástica
- **Boltzmann** - Selección de Boltzmann
- **Boltzmann Annealing** - Selección de Boltzmann con recocido simulado

### Algoritmos de cruce implementados
- **One Point** - Cruce de un punto
- **Two Points** - Cruce de dos puntos
- **Limited Multigene** - Cruce multigénico limitado

### Algoritmos de mutación implementados
- **Single** - Mutación de un gen
- **Multigene** - Mutación multigénica

### Ejecutar análisis completo con todas las configuraciones
#### Configuración
Editar archivos en `configs/` para personalizar:
- Imagen objetivo
- Tamaño de población inicial
- Número de triángulos por solución
- Probabilidades de cruce y mutación
- Algoritmos de selección, cruce y mutación
- Criterios de parada (generaciones, tiempo, fitness)

#### Ejecución
```shell
# Entrar al directorio de scripts
cd scripts

# Ejecutar todas las configuraciones en paralelo
python run_all_configs.py

# Ejecutar con número específico de tareas paralelas
python run_all_configs.py -j 8

# Ejecutar con salida detallada
python run_all_configs.py --verbose
```

### Configuraciones predefinidas
El proyecto incluye múltiples configuraciones en la carpeta `configs/`:

**Configuraciones de selección:**
- `boltzmann_annealing.json` - Boltzmann con recocido simulado
- `boltzmann_high_temp.json` - Boltzmann con temperatura alta
- `boltzmann_low_temp.json` - Boltzmann con temperatura baja
- `elite.json` - Selección elitista
- `ranking.json` - Selección por ranking
- `roulette.json` - Selección por ruleta
- `tournament_deterministic.json` - Torneo determinístico
- `tournament_probabilistic_high_threshold.json` - Torneo probabilístico con umbral alto
- `tournament_probabilistic_low_threshold.json` - Torneo probabilístico con umbral bajo
- `universal.json` - Selección universal estocástica

**Configuraciones de cruce:**
- `one_point.json` - Cruce de un punto
- `two_points.json` - Cruce de dos puntos
- `limited_multigene.json` - Cruce multigénico limitado

**Configuraciones de mutación:**
- `single_gene.json` - Mutación de un gen

**Configuraciones especiales:**
- `traditional.json` - Configuración tradicional
- `young_bias.json` - Sesgo hacia individuos jóvenes

### Visualizar resultados
Los resultados se generan automáticamente en la carpeta `outputs/`:

**Para cada configuración ejecutada:**
- `solution.png` - Imagen final aproximada
- `metrics.json` - Métricas de ejecución (fitness, generaciones, tiempo, etc.)
- `progress/` - Imágenes de progreso por generación (si está habilitado)

**Métricas incluidas:**
- Fitness final alcanzado
- Número de generaciones ejecutadas
- Tiempo de ejecución en segundos
- Razón de parada del algoritmo
- Métricas por generación

### Estructura del proyecto
```
tp2_genetics/
├── main.py                    # Punto de entrada principal
├── config.json               # Configuración por defecto
├── images/                   # Imágenes objetivo
│   ├── jamaica.jpg
│   └── noche_estrellada.jpg
├── configs/                  # Configuraciones predefinidas
├── genetic_algorithm/        # Código fuente del algoritmo
│   ├── algorithm_definition.py
│   ├── crossover.py
│   ├── fitness.py
│   ├── mutation.py
│   ├── models/              # Modelos de datos
│   ├── selection_algorithms/ # Algoritmos de selección
│   └── utils/               # Utilidades
├── scripts/                 # Scripts de análisis
│   └── run_all_configs.py  # Ejecutor de todas las configuraciones
└── outputs/                 # Resultados generados
```

### Parámetros de configuración
- `target_image_path` - Ruta a la imagen objetivo
- `initial_population_size` - Tamaño de la población inicial
- `triangles_per_solution` - Número de triángulos por solución
- `recombination_probability` - Probabilidad de cruce
- `mutation_probability` - Probabilidad de mutación
- `mutation_delta_max_percent` - Porcentaje máximo de cambio en mutación
- `parents_selection_percentage` - Porcentaje de padres seleccionados
- `gen_cutoff` - Límite de generaciones
- `fitness_cutoff` - Límite de fitness
- `minutes_cutoff` - Límite de tiempo en minutos
- `no_change_gens_cutoff` - Generaciones sin cambio para parar
- `young_bias` - Sesgo hacia individuos jóvenes
- `progress_saving` - Guardar progreso por generación
