# API modelo de datos Spike


## 💻 Pré-requisitos

Antes de começar, verifique se você atendeu aos seguintes requisitos:

* Tener instalado docker en tu computador y ademas docker-composer para cargar el contenedor de pruebas con AIRDFLOW.
* Python3.7 o superior.
* Insomnia/Postman o la extención e Rest API de VisualStudioCode. En este readme lo demosstraremos con INSOMNIA, el cual puedes descargar 
[de este link](https://insomnia.rest/download).*()


### Para realizar pruebas con la API, puedes utilizar Insomnia/Postman. 






curl -i -X POST  http://127.0.0.1:3000/spike/model_1/csv  -H "Content-Type: text/csv" --data-binary "@./data/x_test.csv"


