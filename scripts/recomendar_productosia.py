import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def cargar_datos():
    clientes_df = pd.read_csv('data/clientes.csv')
    compras_df = pd.read_csv('data/compras.csv')
    inventario_df = pd.read_csv('data/inventario.csv')
    return clientes_df, compras_df, inventario_df

def recomendar_productos(cliente_id, compras_df, inventario_df):
    # Unir compras_df con inventario_df para obtener la columna de grupo
    compras_df = compras_df.merge(inventario_df[['id_producto', 'grupo']], on='id_producto', how='left')

    # Crear una matriz de conteo de grupos de productos comprados por cliente
    compras_df['cantidad'] = 1
    matriz_grupos = compras_df.pivot_table(index='id_cliente', columns='grupo', values='cantidad', aggfunc='sum', fill_value=0)

    # Verificar si el cliente_id está en la matriz_grupos
    if cliente_id not in matriz_grupos.index:
        print(f"Cliente ID {cliente_id} no encontrado en la matriz de grupos.")
        return []

    # Calcular la similitud de coseno entre clientes
    matriz_similitud = cosine_similarity(matriz_grupos)

    # Encontrar clientes similares
    cliente_indice = matriz_grupos.index.get_loc(cliente_id)
    similitud_cliente = matriz_similitud[cliente_indice]

    # Recomendación de grupos basada en clientes similares
    recomendacion_grupos = matriz_grupos.T.dot(similitud_cliente)
    recomendacion_grupos = recomendacion_grupos.sort_values(ascending=False)

    # Obtener los grupos comprados por el cliente
    grupos_comprados = matriz_grupos.loc[cliente_id]

    # Filtrar grupos ya comprados por el cliente
    grupos_no_comprados = recomendacion_grupos[grupos_comprados == 0]

    # Obtener los productos recomendados de los grupos no comprados
    productos_recomendados = inventario_df[inventario_df['grupo'].isin(grupos_no_comprados.index)].drop_duplicates(subset='id_producto')

    return productos_recomendados.head(10)

if __name__ == "__main__":
    clientes_df, compras_df, inventario_df = cargar_datos()
    cliente_id = 1  # ID del cliente para el cual generar recomendaciones
    productos_recomendados = recomendar_productos(cliente_id, compras_df, inventario_df)
    print("Productos recomendados para el cliente", cliente_id)
    print(productos_recomendados)