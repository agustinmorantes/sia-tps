## TP4 - Unsupervised Learning:
## Ejercicio 1 — Kohonen (Clustering de países europeos)
Directorio: `exercise_1`

1) Ejecutar el entrenamiento y clustering:
```bash
cd exercise_1
python main.py
```

Esto:
- Entrena un SOM con los datos de `europe.csv`.
- Imprime los clusters por neurona ganadora.
- Guarda resultados en `clustering_results_YYYYMMDD_HHMMSS.json` y `clustering_results_latest.json`.

2) Visualizaciones opcionales:
- Visualizar clusters (usa el último resultado por defecto):
```bash
python visualize_clusters.py
# o con un archivo específico
python visualize_clusters.py clustering_results_YYYYMMDD_HHMMSS.json
```
- U-Matrix / Heatmaps:
```bash
python u_matrix_visualizer.py
python feature_heatmaps.py
```

---

## Ejercicio 2 — PCA con Regla de Oja y comparación de perceptrones
Directorio: `exercise_2`

1) PCA (Oja):
```bash
cd exercise_2
python oja_pca.py
```

Esto:
- Estandariza `europe.csv` y entrena un perceptrón lineal con la regla de Oja.
- Proyecta los datos en la 1ra componente principal.
- Genera `pca1_per_country.png` y muestra métricas/cargas en consola.

---

## Ejercicio 3 — Hopfield
Directorio: `exercise_3`

1) Ejecutar con un archivo de configuración:
```bash
cd exercise_3
python main.py configs/a_low_noise.json
```

2) Ejecutar todos los configuraciones del directorio `configs/`:
```bash
python run_all.py
```

Salidas:
- Métricas por configuración en `metrics/<config>.json`.
- Gráficos consolidados en `output_graphs/`.

---

## Extra — Análisis PCA con scikit-learn
Directorio: `exercise_pca`

```bash
cd exercise_pca
python main.py
```

Esto carga `../exercise_1/europe.csv`, estandariza, ejecuta PCA de `sklearn`, imprime análisis de PC1 y genera visualizaciones (biplot, boxplots, barras).
