import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class EuropeanDataAnalyzer:
    
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.countries = []
        self.features = []
        self.feature_names = []
        self.scaled_data = None
        self.pca_model = None
        self.transformed_data = None
        
    def load_european_data(self):
        print("Cargando datos")

        with open(self.data_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            self.feature_names = next(csv_reader)[1:] # Excluir 'Country'
            
            for row in csv_reader:
                self.countries.append(row[0])
                self.features.append([float(x) for x in row[1:]])
        
        # Convertir a numpy array para operaciones
        self.features = np.array(self.features)
        return self
    
    def preprocess_data(self):
        print("Preprocesando datos")
        
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.features)
        return self
    
    def perform_pca_analysis(self, n_components=7):        
        self.pca_model = PCA(n_components=n_components)
        self.transformed_data = self.pca_model.fit_transform(self.scaled_data)
        
        print("Análisis PCA completado")
        return self
    
    def analyze_first_component(self):
        print("Analizando PC1")
        
        variance_explained = self.pca_model.explained_variance_ratio_[0]
        print(f"Varianza explicada por PC1: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
        
        print(f"\nCargas de PC1:")
        for i, feature in enumerate(self.feature_names):
            loading = self.pca_model.components_[0][i]
            print(f"{feature}: {loading:.4f}")
        
        # Variables más importantes
        print(f"\nVariables ordenadas por importancia en PC1:")
        loadings_pc1 = self.pca_model.components_[0]
        abs_loadings = np.abs(loadings_pc1)
        sorted_indices = np.argsort(abs_loadings)[::-1]
        
        for i, idx in enumerate(sorted_indices):
            feature_name = self.feature_names[idx]
            loading = loadings_pc1[idx]
            print(f"{feature_name}: {loading:.4f}")
        
        # Países extremos
        pc1_scores = self.transformed_data[:, 0]
        max_country_idx = np.argmax(pc1_scores)
        min_country_idx = np.argmin(pc1_scores)
        
        print(f"\nPaíses extremos en PC1:")
        print(f"Mayor valor: {self.countries[max_country_idx]} ({pc1_scores[max_country_idx]:.4f})")
        print(f"Menor valor: {self.countries[min_country_idx]} ({pc1_scores[min_country_idx]:.4f})")    
        return pc1_scores
    
    def create_visualizations(self, pc1_scores):
        print("Generando gráficos")
        
        # Boxplot de datos originales
        self._create_raw_data_boxplot()
        
        # Boxplot de datos normalizados
        self._create_normalized_boxplot()

         # Biplot
        self._create_biplot()
        
        # Gráfico de barras PC1
        self._create_pc1_bar_chart(pc1_scores)
    
    def _create_biplot(self):
        plt.figure(figsize=(12, 10))
        
        # Scatter plot de países
        plt.scatter(self.transformed_data[:, 0], self.transformed_data[:, 1], 
                   s=100, alpha=0.7, c='blue')
        
        # Etiquetas de países
        for i, country in enumerate(self.countries):
            plt.annotate(country, (self.transformed_data[i, 0], self.transformed_data[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Vectores de variables
        scale_factor = 3
        for i, feature in enumerate(self.feature_names):
            plt.arrow(0, 0, 
                     self.pca_model.components_[0][i] * scale_factor,
                     self.pca_model.components_[1][i] * scale_factor,
                     color='red', alpha=0.7, width=0.01)
            plt.text(self.pca_model.components_[0][i] * scale_factor,
                    self.pca_model.components_[1][i] * scale_factor,
                    feature, color='red', fontsize=10, ha='center')
        
        plt.xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Biplot: Países y Variables en el espacio PC1-PC2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _create_normalized_boxplot(self):
        plt.figure(figsize=(12, 8))
        
        # Normalizar manualmente para el boxplot
        normalized_data = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0)
        
        box_plot = plt.boxplot(normalized_data, patch_artist=True, tick_labels=self.feature_names)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                 '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        plt.xticks(rotation=45)
        plt.title('Boxplot de Datos Normalizados')
        plt.ylabel('Valores Normalizados')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _create_raw_data_boxplot(self):
        plt.figure(figsize=(12, 8))
        
        box_plot = plt.boxplot(self.features, patch_artist=True, tick_labels=self.feature_names)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                 '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        plt.xticks(rotation=45)
        plt.title('Boxplot de Datos Originales')
        plt.ylabel('Valores Originales')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _create_pc1_bar_chart(self, pc1_scores):
        plt.figure(figsize=(15, 8))
        
        # Ordenar países por PC1
        sorted_indices = np.argsort(pc1_scores)
        sorted_countries = [self.countries[i] for i in sorted_indices]
        sorted_scores = [pc1_scores[i] for i in sorted_indices]
        
        # Colores: verde para valores positivos, rojo para negativos
        colors = ['green' if score >= 0 else 'red' for score in sorted_scores]
        
        bars = plt.bar(range(len(sorted_countries)), sorted_scores, color=colors, alpha=0.7)
        
        plt.xlabel('Países')
        plt.ylabel('Índice PC1')
        plt.title('Índice de Desarrollo Económico-Social (PC1) por País')
        plt.xticks(range(len(sorted_countries)), sorted_countries, rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        
        (self.load_european_data()
         .preprocess_data()
         .perform_pca_analysis())
        
        # Análisis de PC1
        pc1_scores = self.analyze_first_component()
        
        # Visualizaciones
        self.create_visualizations(pc1_scores)        
        return self


def main():
    data_file = '../exercise_1/europe.csv'
    
    # Crear y ejecutar analizador
    analyzer = EuropeanDataAnalyzer(data_file)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()