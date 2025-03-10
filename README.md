# Soccer Data DBC

Un dashboard interactivo para análisis deportivo desarrollado con Dash y Python.


## Descripción

Soccer Data DBC es una aplicación web para el análisis de rendimiento de jugadores de fútbol que permite visualizar estadísticas, comparar jugadores y generar informes personalizados. La plataforma integra datos de múltiples fuentes y ofrece visualizaciones interactivas para facilitar el análisis deportivo.

## Características

- **Sistema de autenticación** seguro con Flask-Login
- **Múltiples dashboards especializados**:
  - Dashboard principal con estadísticas generales
  - Búsqueda de jugadores similares
  - Análisis de rendimiento defensivo
  - Análisis de rendimiento ofensivo
  - Análisis de creación de juego
  - Análisis de superación de oponentes
- **Visualizaciones interactivas** con Plotly
- **Filtros avanzados** para personalizar el análisis
- **Exportación a PDF** de informes personalizados
- **Diseño responsivo** adaptado a diferentes dispositivos

## Requisitos

- Python 3.7+
- Bibliotecas listadas en requirements.txt

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tu-usuario/soccer-data-dbc.git
cd soccer-data-dbc
```

2. Instala las dependencias requeridas:
```bash
pip install -r requirements.txt
```

3. Asegúrate de tener los archivos de datos en la carpeta `data/`:
   - estadisticas_limpio.xlsx
   - estadisticas_final.xlsx

## Uso

1. Ejecuta la aplicación:
```bash
python index.py
```

2. Accede a la interfaz web en tu navegador:
```
http://127.0.0.1:8050/
```

3. Inicia sesión con las siguientes credenciales:
   - Usuario: `admin`
   - Contraseña: `admin`

## Estructura del Proyecto

```
soccer-data-dbc/
│
├── app.py                     # Configuración principal de la aplicación
├── index.py                   # Punto de entrada y manejo de rutas
├── assets/                    # Archivos estáticos (CSS, imágenes)
│   ├── logos_ligas/           # Logos de las ligas
│   └── players/               # Imágenes de jugadores
│
├── components/                # Componentes reutilizables
│   ├── data_manager.py        # Gestor de datos (patrón Singleton)
│   ├── pdf_export.py          # Funciones para exportar a PDF
│   └── sidebar.py             # Componente de barra lateral
│
├── data/                      # Archivos de datos
│   ├── estadisticas_limpio.xlsx
│   └── estadisticas_final.xlsx
│
└── pages/                     # Páginas de la aplicación
    ├── login.py               # Página de inicio de sesión
    ├── dashboard.py           # Dashboard principal
    ├── jugadores_similares.py # Búsqueda de jugadores similares
    ├── rendimiento_defensivo.py
    ├── rendimiento_ofensivo.py
    ├── creacion_juego.py
    └── analisis_superacion.py
```

## Funcionalidades Detalladas

### Dashboard Principal
Muestra una visión general de los datos, incluyendo distribución por posiciones, ligas, y estadísticas generales.

### Jugadores Similares
Permite encontrar jugadores con características similares a un jugador base seleccionado, utilizando análisis multidimensional y métricas ponderadas.

### Rendimiento Defensivo
Visualiza y analiza métricas defensivas como duelos ganados, intercepciones y recuperaciones.

### Rendimiento Ofensivo
Analiza la efectividad ofensiva en términos de goles, tiros y expected goals (xG).

### Creación de Juego
Visualiza métricas relacionadas con la creación de juego como pases, asistencias y zonas de influencia.

### Análisis de Superación
Analiza la capacidad de los jugadores para superar oponentes a través de regates y conducción.

## Exportación a PDF

Cada dashboard incluye la capacidad de exportar informes a PDF, personalizando el título, descripción y filtros aplicados.

## Créditos

Desarrollado como parte del Máster en Python Avanzado Aplicado al Deporte.

## Licencia

Este proyecto está licenciado bajo los términos de la Licencia MIT. Ver el archivo LICENSE para más detalles.