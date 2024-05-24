import pandas as pd  # Importa la biblioteca pandas con el alias pd para manipulación y análisis de datos.

def cargar_datos():
    # Función para cargar los datos de clientes, compras e inventario desde archivos CSV.
    clientes_df = pd.read_csv('data/clientes.csv')  # Lee el archivo clientes.csv y carga los datos en un DataFrame utilizando pandas.
    compras_df = pd.read_csv('data/compras.csv')  # Lee el archivo compras.csv y carga los datos en un DataFrame utilizando pandas.
    inventario_df = pd.read_csv('data/inventario.csv')  # Lee el archivo inventario.csv y carga los datos en un DataFrame utilizando pandas.
    return clientes_df, compras_df, inventario_df  # Retorna los DataFrames cargados.

if __name__ == "__main__":
    # Si el script es ejecutado como un programa principal, cargar los datos y mostrar los primeros registros de cada DataFrame.
    clientes_df, compras_df, inventario_df = cargar_datos()
    print("Clientes:")
    print(clientes_df.head())
    print("\nCompras:")
    print(compras_df.head())
    print("\nInventario:")
    print(inventario_df.head())
