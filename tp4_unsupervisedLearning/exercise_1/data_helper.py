import numpy as np

def standardize_data(data_matrix): #(Z-Score)


    means = np.mean(data_matrix, axis=0) # Calcular media  para cada columna
    stds = np.std(data_matrix, axis=0, ddof=0)   # Calcular el desvio standar para cada columna
    stds_safe = np.where(stds == 0, 1, stds)  # Manejar desvío estándar cero (evitar división por cero)
    standardized_data = (data_matrix - means) / stds_safe  # Aplicar la variable estadarizada
    return standardized_data, means, stds

def load_and_standardize_europe_data(filepath): #Carga y estandariza los datos de Europa desde el CSV
   
    data = []
    countries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:] 
        for line in lines:
            line = line.strip()
            if line:  
                parts = line.split(',')
                country = parts[0].strip('"')
                values = [float(x) for x in parts[1:]]  
                countries.append(country)
                data.append(values)
    X = np.array(data)
    X_standardized, means, stds = standardize_data(X) # Estandarizar datos
    return X_standardized, countries, means, stds
