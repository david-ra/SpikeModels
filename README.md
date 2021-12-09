# API modelo de datos Spike
El siguiente repositorio tiene como finalidad exponer una API que pueda utilizarse para la predicción de precios de insumos básicos en Chile.
La API fue construida en base a un archivo JupyterNotebook que implementaba la fase de preprocesamiento de datos, entrenamiento y testeo de modelos.
El repositorio incluye un archivo docker-compose para levantar un ambiente de Airflow, para testear pipeline de entrenamiento y generacion de modelos, 
en el caso que se requiera realizar un reentrenamiento del modelo a futuro con datos actualizados.


## 💻 Pré-requisitos

Antes de começar, verifique se você atendeu aos seguintes requisitos:

* Tener instalado docker en tu computador y ademas docker-composer para cargar el contenedor de pruebas con AIRDFLOW.
* Python3.7 o superior.
* Insomnia/Postman o la extención e Rest API de VisualStudioCode. En este readme lo demostraremos con INSOMNIA, el cual puedes descargar 
[de este link](https://insomnia.rest/download).
* Si usas linux, y cuentas con CURL instalado, tambien pasaremos el comando para realizar test con la api del modelo.


## Creando contenedor para testeo de API.

Primero debemos clonar el repositorio de la API, y dentro del directorio SpikeModels ejecutar:

`docker-compose up -d`

Esto expondra el container llamado "spike_model" en el puerto 5000. Si deseas cambiar el puerto, 
puedes hacerlo redirigiento otro que tengas libre dentro del docker-compose.yml en la sección ports.

Dentro del repositorio existe una carpeta llama data el cual contiene los siguientes archivos:

```
SpikeModels/
 | data/
 |   | banco_central.csv
 |   | precio_leche.csv
 |   | precipitaciones.csv
 |   | x_test.csv
```
los 3 primeros csv son los archivos csv de datos utilizados para preprocesamiento y entrenamiento, el archivo x_test.csv es un archivo 
utilizado como input de testeo, que corresponde al 20% de los datos que no se utilizaron para entrenar, con el fin de que puedas
testear inmediatamente el output de la API enviando ese archivo mediante POST con INSOMNIA ( o Postman si lo prefieres ).

## Para testear usando INSOMNIA

abrir insomnia y creamos un nuevo request:

<img src="create_request.png" alt="ejemplo crear request">




### Para realizar pruebas con la API, puedes utilizar Insomnia/Postman. 




curl -i -X POST  http://127.0.0.1:3000/spike/model_1/csv  -H "Content-Type: text/csv" --data-binary "@./data/x_test.csv"


