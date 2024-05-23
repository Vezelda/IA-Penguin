import pandas as pd

def cargar_datos():
    clientes_df = pd.read_csv('data/clientes.csv')
    compras_df = pd.read_csv('data/compras.csv')
    inventario_df = pd.read_csv('data/inventario.csv')
    return clientes_df, compras_df, inventario_df

if __name__ == "__main__":
    clientes_df, compras_df, inventario_df = cargar_datos()
    print("Clientes:")
    print(clientes_df.head())
    print("\nCompras:")
    print(compras_df.head())
    print("\nInventario:")
    print(inventario_df.head())
print(clientes_df)
