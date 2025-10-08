import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_training_history(history, save_path=None):
    """Plotea el historial de entrenamiento"""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()

def plot_confusion_matrix(confusion_matrix, save_path=None):
    """Plotea la matriz de confusión"""
    plt.figure(figsize=(10, 8))
    
    # Normalizar la matriz
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Crear heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    
    plt.title('Matriz de Confusión Normalizada')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()

def plot_class_metrics(precision, recall, f1_score, save_path=None):
    """Plotea las métricas por clase"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    classes = range(10)
    
    # Precision
    ax1.bar(classes, precision, color='skyblue', alpha=0.7)
    ax1.set_title('Precision por Clase')
    ax1.set_xlabel('Clase')
    ax1.set_ylabel('Precision')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Recall
    ax2.bar(classes, recall, color='lightcoral', alpha=0.7)
    ax2.set_title('Recall por Clase')
    ax2.set_xlabel('Clase')
    ax2.set_ylabel('Recall')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # F1-Score
    ax3.bar(classes, f1_score, color='lightgreen', alpha=0.7)
    ax3.set_title('F1-Score por Clase')
    ax3.set_xlabel('Clase')
    ax3.set_ylabel('F1-Score')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Métricas por clase guardadas en: {save_path}")
    
    plt.show()

def plot_sample_predictions(X_test, y_test, y_pred, n_samples=10, save_path=None):
    """Plotea muestras con predicciones"""
    # Convertir one-hot a clases
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Reshape imagen a 28x28
        image = X_test[idx].reshape(28, 28)
        
        # Plot imagen
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {y_test_classes[idx]}, Pred: {y_pred_classes[idx]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Muestras de predicción guardadas en: {save_path}")
    
    plt.show()

def plot_architecture_comparison(results_dict, save_path=None):
    """Compara diferentes arquitecturas"""
    architectures = list(results_dict.keys())
    accuracies = [results_dict[arch]['test_accuracy'] for arch in architectures]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(architectures, accuracies, color='lightblue', alpha=0.7)
    
    # Añadir valores en las barras
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.title('Comparación de Arquitecturas')
    plt.xlabel('Arquitectura')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparación de arquitecturas guardada en: {save_path}")
    
    plt.show()

def plot_learning_curves_comparison(histories_dict, save_path=None):
    """Compara curvas de aprendizaje de diferentes configuraciones"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for name, history in histories_dict.items():
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, history['train_loss'], label=f'{name} - Train', linestyle='-')
        
        # Accuracy
        ax2.plot(epochs, history['train_acc'], label=f'{name} - Train', linestyle='-')
    
    ax1.set_title('Comparación de Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Comparación de Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparación de curvas guardada en: {save_path}")
    
    plt.show()
