import numpy as np
import os

class PatternLoader: #Carga patrones desde archivos .txt 
  
    
    def __init__(self):
        self.patterns = []  # Lista de patrones almacenados como vectores
        self.pattern_names = []  # Nombres de los patrones
        self.pattern_matrices = []  # Lista de patrones como matrices 5x5
    
    def load_pattern_from_file(self, file_path):

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Filtrar líneas vacías y procesar cada línea
            matrix_rows = []
            for line in lines:
                line = line.strip()
                if line:  # Si la línea no está vacía
                    # Dividir por espacios y convertir a enteros
                    row = [int(x) for x in line.split()]
                    if len(row) == 5:  # Verificar que tenga exactamente 5 elementos
                        matrix_rows.append(row)
            
            if len(matrix_rows) != 5:
                raise ValueError(f"El archivo debe contener exactamente 5 filas, se encontraron {len(matrix_rows)}")
            
            # Convertir a matriz numpy
            pattern_matrix = np.array(matrix_rows)
            
            # Verificar que todos los valores sean 1 o -1
            if not np.all((pattern_matrix == 1) | (pattern_matrix == -1)):
                raise ValueError("Todos los valores deben ser 1 o -1")
            
            return pattern_matrix
            
        except FileNotFoundError:
            raise FileNotFoundError(f"No se pudo encontrar el archivo: {file_path}")
        except Exception as e:
            raise Exception(f"Error al cargar el archivo {file_path}: {str(e)}")
    
    def matrix_to_vector(self, matrix):
   
        return matrix.flatten()  # Convierte por filas (row-major order)
    
    def vector_to_matrix(self, vector):
    
        return vector.reshape(5, 5)
    
    def load_stored_patterns(self, resources_dir="Resources", selected_patterns=None):
     
        if selected_patterns is None:
            
            selected_patterns = ['a.txt', 't.txt', 'x.txt', 'j.txt']
        
        for pattern_file in selected_patterns:
            file_path = os.path.join(resources_dir, pattern_file)
            pattern_name = pattern_file.replace('.txt', '').upper()
            
            try:
                # Cargar patrón como matriz 5x5
                pattern_matrix = self.load_pattern_from_file(file_path)
                
                # Convertir a vector
                pattern_vector = self.matrix_to_vector(pattern_matrix)
                
                # Almacenar tanto como matriz como vector
                self.pattern_matrices.append(pattern_matrix)
                self.patterns.append(pattern_vector)
                self.pattern_names.append(pattern_name)
                
            except Exception as e:
                print(f"Error cargando patrón {pattern_name}: {e}")
    
    def visualize_pattern(self, pattern, pattern_name="Pattern"):
      
        # Convertir a matriz si es necesario
        if pattern.ndim == 1:
            matrix = self.vector_to_matrix(pattern)
        else:
            matrix = pattern
        
        print(f"\n{pattern_name}:")
        print("_" * 7)
        for row in matrix:
            line = ""
            for pixel in row:
                if pixel == 1:
                    line += " *"
                else:  # pixel == -1
                    line += "  "
            print(line)
        print("_" * 7)
    
    def show_all_patterns(self):
       
        if not self.patterns:
            print("No hay patrones cargados.")
            return
        
        print("\n" + "="*50)
        print("PATRONES ALMACENADOS")
        print("="*50)
        
        for i, (pattern, name) in enumerate(zip(self.patterns, self.pattern_names)):
            self.visualize_pattern(pattern, f"Patrón {i+1} - Letra {name}")
    
    def get_patterns_as_vectors(self):
     
        return self.patterns.copy()
    
    def get_patterns_as_matrices(self):
      
        return self.pattern_matrices.copy()
    
    def get_pattern_names(self):
       
        return self.pattern_names.copy()

def main():
    
    print("CARGADOR DE PATRONES - PRUEBA")
    print("="*40)
    
    # Crear instancia del cargador
    loader = PatternLoader()
    
    # Cargar patrones
    loader.load_stored_patterns()
    
    # Mostrar patrones
    loader.show_all_patterns()
    
    return loader

if __name__ == "__main__":
    loader = main()
