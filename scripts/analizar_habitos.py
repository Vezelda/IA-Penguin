import pandas as pd  # Importa la biblioteca pandas con el alias pd para manipulación y análisis de datos.
import matplotlib.pyplot as plt  # Importa el módulo pyplot de la biblioteca matplotlib con el alias plt para crear gráficos.

def cargar_datos():
    # Cargar los datos de clientes, compras e inventario desde archivos CSV.
    clientes_df = pd.read_csv('data/clientes.csv')  # Lee el archivo clientes.csv y lo carga en un DataFrame utilizando pandas.
    compras_df = pd.read_csv('data/compras.csv')  # Lee el archivo compras.csv y lo carga en un DataFrame utilizando pandas.
    inventario_df = pd.read_csv('data/inventario.csv')  # Lee el archivo inventario.csv y lo carga en un DataFrame utilizando pandas.
    return clientes_df, compras_df, inventario_df

def analizar_habitos(clientes_df, compras_df, inventario_df):
    # Función para analizar los hábitos de compra de los clientes.
    frecuencia_compra = compras_df['id_cliente'].value_counts()  # Calcula la frecuencia de compra por cliente.
    valor_medio_compra = compras_df.groupby('id_cliente')['cantidad'].mean()  # Calcula el valor medio de compra por cliente.
    productos_populares = compras_df['id_producto'].value_counts()  # Determina los productos más populares basados en la cantidad vendida.
    return frecuencia_compra, valor_medio_compra, productos_populares

def visualizar_datos(frecuencia_compra, valor_medio_compra, productos_populares):
    # Función para visualizar los datos analizados.
    plt.figure(figsize=(10, 6))

    # Histograma de la frecuencia de compra por cliente.
    plt.subplot(3, 1, 1)
    plt.hist(frecuencia_compra, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Frecuencia de Compra')
    plt.ylabel('Número de Clientes')
    plt.title('Distribución de Frecuencia de Compra')

    # Histograma del valor medio de compra por cliente.
    plt.subplot(3, 1, 2)
    plt.hist(valor_medio_compra, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Valor Medio de Compra')
    plt.ylabel('Número de Clientes')
    plt.title('Distribución de Valor Medio de Compra')

    # Gráfico de barras de los productos más populares.
    plt.subplot(3, 1, 3)
    productos_populares.head(10).plot(kind='bar', color='coral', edgecolor='black')
    plt.xlabel('ID del Producto')
    plt.ylabel('Cantidad Vendida')
    plt.title('Productos Más Populares')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    clientes_df, compras_df, inventario_df = cargar_datos()  # Carga los datos utilizando la función cargar_datos.
    frecuencia_compra, valor_medio_compra, productos_populares = analizar_habitos(clientes_df, compras_df, inventario_df)  # Realiza el análisis de hábitos de compra.
    visualizar_datos(frecuencia_compra, valor_medio_compra, productos_populares)  # Visualiza los resultados del análisis.


