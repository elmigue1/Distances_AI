# Proyecto CRUD de Usuarios en Python

Este proyecto es una implementación CRUD (Crear, Leer, Actualizar, Eliminar) de la entidad "Usuario", desarrollada en Python utilizando **Flask** como framework web. Permite gestionar la información de los usuarios y sus datos asociados en una base de datos relacional, y presenta los datos a través de una interfaz web sencilla.

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener las siguientes herramientas instaladas:

- **Python** (versión 3.6 o superior)

## Instrucciones de instalación

### 1. Crear un entorno virtual

Primero, crea un entorno virtual para el proyecto. Asegúrate de tener Python instalado y luego ejecuta los siguientes comandos:

#### En Windows:
```bash
python -m venv venv
venv\Scripts\activate

#### En Mac/Linux:
```bash
python -m venv venv
venv\Scripts\activate

### 2. Instalar dependencias:

pip install -r requirements.txt

### 3. Ejecutar aplicación:
Una vez que hayas instalado todas las dependencias y configurado la base de datos, puedes iniciar la aplicación ejecutando el archivo principal:
```bash
python app.py

### 4. Resultados:

Una vez que la aplicación esté en ejecución, podrás acceder a la interfaz web. Los resultados de la aplicación se verán en la ruta /usuarios_html dentro del directorio donde hayas guardado la aplicación.
