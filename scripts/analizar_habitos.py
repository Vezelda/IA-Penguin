import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos():
    clientes_df = pd.read_csv('data/clientes.csv')
    compras_df = pd.read_csv('data/compras.csv')
    inventario_df = pd.read_csv('data/inventario.csv')
    return clientes_df, compras_df, inventario_df

def analizar_habitos(clientes_df, compras_df, inventario_df):
    # Frecuencia de compra por cliente
    frecuencia_compra = compras_df['id_cliente'].value_counts()

    # Valor medio de compra por cliente
    valor_medio_compra = compras_df.groupby('id_cliente')['cantidad'].mean()

    # Productos más populares
    productos_populares = compras_df['id_producto'].value_counts()

    return frecuencia_compra, valor_medio_compra, productos_populares

def visualizar_datos(frecuencia_compra, valor_medio_compra, productos_populares):
    plt.figure(figsize=(10, 6))

    # Frecuencia de compra
    plt.subplot(3, 1, 1)
    plt.hist(frecuencia_compra, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Frecuencia de Compra')
    plt.ylabel('Número de Clientes')
    plt.title('Distribución de Frecuencia de Compra')

    # Valor medio de compra
    plt.subplot(3, 1, 2)
    plt.hist(valor_medio_compra, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Valor Medio de Compra')
    plt.ylabel('Número de Clientes')
    plt.title('Distribución de Valor Medio de Compra')

    # Productos populares
    plt.subplot(3, 1, 3)
    productos_populares.head(10).plot(kind='bar', color='coral', edgecolor='black')
    plt.xlabel('ID del Producto')
    plt.ylabel('Cantidad Vendida')
    plt.title('Productos Más Populares')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    clientes_df, compras_df, inventario_df = cargar_datos()
    frecuencia_compra, valor_medio_compra, productos_populares = analizar_habitos(clientes_df, compras_df, inventario_df)
    visualizar_datos(frecuencia_compra, valor_medio_compra, productos_populares)
