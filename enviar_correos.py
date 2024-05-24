import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from jinja2 import Template

# Función para cargar datos desde archivos CSV
def cargar_datos():
    clientes_df = pd.read_csv('data/clientes.csv')
    recomendaciones_df = pd.read_csv('data/recomendaciones.csv')
    return clientes_df, recomendaciones_df


# Función para enviar correo electrónico
def enviar_correo(cliente_nombre, cliente_email, productos_recomendados):
    from_email = "iapenguinhackaton@gmail.com"
    from_password = "tdtd nkff nxyy pjjs"
    subject = "Tu proxima compra esta aqui!"
    with open('templates/email_template.html') as file_:
        template = Template(file_.read())
    # Saludo personalizado con el nombre del cliente
    saludo = f"Hola {cliente_nombre}! Creemos que te podrían interesar estos productos."
    email_content = template.render(saludo=saludo, productos=productos_recomendados)
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = cliente_email
    msg['Subject'] = subject
    msg.attach(MIMEText(email_content, 'html'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    server.send_message(msg)
    server.quit()

# Función para enviar correos a todos los clientes
def enviar_correos_a_todos_los_clientes(clientes_df, recomendaciones_df):
    for _, cliente in clientes_df.iterrows():
        cliente_id = cliente['id_cliente']
        cliente_nombre = cliente['nombre']
        cliente_email = cliente['correo_electronico']
        productos_recomendados = recomendaciones_df[recomendaciones_df['id_cliente'] == cliente_id]['recomendaciones'].values[0]
        enviar_correo(cliente_nombre, cliente_email, productos_recomendados.split(','))
        print(f"Correo enviado a {cliente_nombre} ({cliente_email})")

if __name__ == "__main__":
    clientes_df, recomendaciones_df = cargar_datos()
    if 'correo_electronico' in clientes_df.columns:
        enviar_correos_a_todos_los_clientes(clientes_df, recomendaciones_df)
    else:
        print("El nombre de la columna 'correo_electronico' no está presente en clientes_df.")

