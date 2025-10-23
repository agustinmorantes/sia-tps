import numpy as np
from pattern_loader import PatternLoader

class HopfieldNetwork:
    def __init__(self):
        self.pattern_loader = PatternLoader()  # Cargador de patrones
        self.patterns = []  # Lista de patrones almacenados como vectores
        self.pattern_names = []  # Nombres de los patrones
        self.weight_matrix = None  # Matriz de pesos W
        
    def load_stored_patterns(self, resources_dir="Resources"):
     
        self.pattern_loader.load_stored_patterns(resources_dir)
        self.patterns = self.pattern_loader.get_patterns_as_vectors()
        self.pattern_names = self.pattern_loader.get_pattern_names()
    
    def show_all_patterns(self):
    
        self.pattern_loader.show_all_patterns()
    
    
    def create_pattern_matrix_K(self): #  Crea la matriz K donde cada columna es un patrón almacenado
       
        if not self.patterns:
            raise ValueError("No hay patrones cargados para crear la matriz K")
        
        # Convertir lista de vectores a matriz donde cada columna es un patrón
        K = np.column_stack(self.patterns)
        
        return K
    
    def compute_weight_matrix_W(self): #    W = (1/N) * K * K^T - I
     
        if not self.patterns:
            raise ValueError("No hay patrones cargados para calcular la matriz de pesos")
        
        # Crear matriz K
        K = self.create_pattern_matrix_K()
        
        N = K.shape[0]  # Número de neuronas
        
        
        W = (1/N) * np.dot(K, K.T)
        
        # Restar matriz identidad - esto pone la diagonal en 0
        I = np.eye(N)
        W = W - I
        
        # Asegurar explícitamente que la diagonal esté en 0
        np.fill_diagonal(W, 0)
        
        # Almacenar la matriz de pesos
        self.weight_matrix = W
        
        return W
    
    def show_complete_weight_matrix(self):
   
        if self.weight_matrix is None:
            print("La matriz de pesos W no ha sido calculada aún.")
            return
        
        W = self.weight_matrix
        print(f"\n" + "="*60)
        print("MATRIZ DE PESOS W COMPLETA (25x25)")
        print("="*60)
        
        # Mostrar la matriz completa con formato
        for i in range(W.shape[0]):
            row_str = ""
            for j in range(W.shape[1]):
                # Formatear cada elemento con 4 decimales
                element = f"{W[i, j]:6.3f}"
                row_str += element + " "
            print(f"Fila {i+1:2d}: {row_str}")
        
        print("="*60)

def main():
 
    print("MODELO DE HOPFIELD - PASOS 1 y 2: Carga y Matriz de Pesos")
    print("="*70)
    
    # Crear instancia de la red
    hopfield = HopfieldNetwork()
    
    # PASO 1: Cargar los patrones seleccionados
    print("\n" + "="*50)
    print("PASO 1: CARGANDO PATRONES")
    print("="*50)
    hopfield.load_stored_patterns()
    hopfield.show_all_patterns()
    
    # PASO 2: Calcular matriz de pesos W
    print("\n" + "="*50)
    print("PASO 2: CALCULANDO MATRIZ DE PESOS W")
    print("="*50)
    W = hopfield.compute_weight_matrix_W()
    
    # Mostrar matriz completa
    hopfield.show_complete_weight_matrix()
    
    return hopfield

if __name__ == "__main__":
    # Ejecutar el paso 1
    network = main()
