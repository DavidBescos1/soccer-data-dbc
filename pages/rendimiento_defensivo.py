from dash import dcc, html, callback, Input, Output, State
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from pdf_export import export_button, create_pdf_report
from components.data_manager import DataManager
from components.sidebar import create_sidebar
from datetime import datetime
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64

# Inicializar el gestor de datos
data_manager = DataManager()

# Definir métricas defensivas principales - Lista completa
metricas_defensivas = [
    "TOQUES DEFENSIVOS ÁREA PROPIA",
    "DUELOS TERRESTRES GANADOS",
    "DUELOS TERRESTRES PERDIDOS",
    "DUELOS AÉREOS GANADOS",
    "DUELOS AÉREOS PERDIDOS",
    "INTERCEPCIONES",
    "BLOQUEOS",
    "RECUPERACIONES BALONES SUELTOS",
    "DESPEJES CABEZA",
    "TOQUES DEFENSIVOS PRIMER TERCIO",
    "RECUPERACIONES PRIMER TERCIO",
    "NÚMERO PRESIONES",
    "TOQUES DEFENSIVOS SIN POSESIÓN",
    "TOQUES TRANSICIÓN DEFENSIVA",
    "RECUPERACIONES SIN POSESIÓN",
    "RECUPERACIONES TRANSICIÓN DEFENSIVA",
    "RECUPERACIONES EN ÚLTIMO TERCIO",
    "VALOR AÑADIDO BLOQUEOS",
    "RECUPERACIONES COMO CENTRAL",
    "RECUPERACIONES COMO LATERAL DERECHO",
    "RECUPERACIONES COMO LATERAL IZQUIERDO",
    "VALOR INTERCEPCIONES EQUIPO",
    "PASES DESDE PRIMER TERCIO",
    "TOQUES DEFENSIVOS EN DUELOS",
    "DUELOS DEFENSIVOS GANADOS",
    "TOQUES DEFENSIVOS CARRIL CENTRAL"
]

# Layout de la página
layout = html.Div([
    # Sidebar
    create_sidebar(),
    
    # Contenido principal
    html.Div(
        className="content-container fade-in",
        children=[
            # Título
            html.H1("Análisis de Rendimiento Defensivo", 
                   style={"color": "#00BFFF", "margin-bottom": "20px"}),
            
            # Filtros
            html.Div(
                className="filter-container",
                children=[
                    # Primera fila de filtros - los 3 dropdowns en la parte superior
                    dbc.Row([
                        # Filtro de posición
                        dbc.Col([
                            html.Div(className="filter-title", children="Posición"),
                            dcc.Dropdown(
                                id="posicion-filter",
                                options=[],  # Se llenará dinámicamente
                                multi=True,
                                placeholder="Selecciona posiciones",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Filtro de liga
                        dbc.Col([
                            html.Div(className="filter-title", children="Liga"),
                            dcc.Dropdown(
                                id="liga-filter",
                                options=[],  # Se llenará dinámicamente
                                multi=True,
                                placeholder="Selecciona ligas",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Filtro de equipos (dependiente de las ligas seleccionadas)
                        dbc.Col([
                            html.Div(className="filter-title", children="Equipo"),
                            dcc.Dropdown(
                                id="equipo-filter",
                                options=[],  # Se llenará dinámicamente según las ligas seleccionadas
                                multi=True,
                                placeholder="Selecciona equipos",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                    ], className="mb-4"),
                    
                    # Segunda fila de filtros - los 2 sliders
                    dbc.Row([
                        # Rango de edad
                        dbc.Col([
                            html.Div(className="filter-title", children="Rango de Edad"),
                            dcc.RangeSlider(
                                id="edad-range-slider",
                                min=16, max=45, step=1,
                                marks={i: str(i) for i in range(16, 46, 5)},
                                value=[18, 35],
                                className="filter-slider"
                            ),
                        ], width=6),
                        
                        # Minutos jugados
                        dbc.Col([
                            html.Div(className="filter-title", children="Minutos Jugados (mínimo)"),
                            dcc.Slider(
                                id="minutos-slider",
                                min=0,
                                max=3000,
                                step=100,
                                value=500,
                                marks={i: str(i) for i in range(0, 3001, 500)},
                                className="filter-slider"
                            ),
                        ], width=6),
                    ], className="mb-4"),
                    
                    # Tercera fila - dropdown de métricas a ancho completo
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="filter-title", children="Métricas Defensivas"),
                            dcc.Dropdown(
                                id="metricas-filter",
                                options=[
                                    {"label": "TOQUES DEFENSIVOS ÁREA PROPIA", "value": "TOQUES DEFENSIVOS ÁREA PROPIA"},
                                    {"label": "DUELOS TERRESTRES GANADOS", "value": "DUELOS TERRESTRES GANADOS"},
                                    {"label": "DUELOS TERRESTRES PERDIDOS", "value": "DUELOS TERRESTRES PERDIDOS"},
                                    {"label": "DUELOS AÉREOS GANADOS", "value": "DUELOS AÉREOS GANADOS"},
                                    {"label": "DUELOS AÉREOS PERDIDOS", "value": "DUELOS AÉREOS PERDIDOS"},
                                    {"label": "INTERCEPCIONES", "value": "INTERCEPCIONES"},
                                    {"label": "BLOQUEOS", "value": "BLOQUEOS"},
                                    {"label": "RECUPERACIONES BALONES SUELTOS", "value": "RECUPERACIONES BALONES SUELTOS"},
                                    {"label": "DESPEJES CABEZA", "value": "DESPEJES CABEZA"},
                                    {"label": "TOQUES DEFENSIVOS PRIMER TERCIO", "value": "TOQUES DEFENSIVOS PRIMER TERCIO"},
                                    {"label": "RECUPERACIONES PRIMER TERCIO", "value": "RECUPERACIONES PRIMER TERCIO"},
                                    {"label": "NÚMERO PRESIONES", "value": "NÚMERO PRESIONES"},
                                    {"label": "TOQUES DEFENSIVOS SIN POSESIÓN", "value": "TOQUES DEFENSIVOS SIN POSESIÓN"},
                                    {"label": "TOQUES TRANSICIÓN DEFENSIVA", "value": "TOQUES TRANSICIÓN DEFENSIVA"},
                                    {"label": "RECUPERACIONES SIN POSESIÓN", "value": "RECUPERACIONES SIN POSESIÓN"},
                                    {"label": "RECUPERACIONES TRANSICIÓN DEFENSIVA", "value": "RECUPERACIONES TRANSICIÓN DEFENSIVA"},
                                    {"label": "RECUPERACIONES EN ÚLTIMO TERCIO", "value": "RECUPERACIONES EN ÚLTIMO TERCIO"},
                                    {"label": "VALOR AÑADIDO BLOQUEOS", "value": "VALOR AÑADIDO BLOQUEOS"},
                                    {"label": "RECUPERACIONES COMO CENTRAL", "value": "RECUPERACIONES COMO CENTRAL"},
                                    {"label": "RECUPERACIONES COMO LATERAL DERECHO", "value": "RECUPERACIONES COMO LATERAL DERECHO"},
                                    {"label": "RECUPERACIONES COMO LATERAL IZQUIERDO", "value": "RECUPERACIONES COMO LATERAL IZQUIERDO"},
                                    {"label": "VALOR INTERCEPCIONES EQUIPO", "value": "VALOR INTERCEPCIONES EQUIPO"},
                                    {"label": "PASES DESDE PRIMER TERCIO", "value": "PASES DESDE PRIMER TERCIO"},
                                    {"label": "TOQUES DEFENSIVOS EN DUELOS", "value": "TOQUES DEFENSIVOS EN DUELOS"},
                                    {"label": "DUELOS DEFENSIVOS GANADOS", "value": "DUELOS DEFENSIVOS GANADOS"},
                                    {"label": "TOQUES DEFENSIVOS CARRIL CENTRAL", "value": "TOQUES DEFENSIVOS CARRIL CENTRAL"}
                                ],
                                value=["DUELOS TERRESTRES GANADOS", "INTERCEPCIONES", "RECUPERACIONES BALONES SUELTOS"],  # Valores por defecto
                                multi=True,
                                placeholder="Selecciona métricas",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Botón de aplicar filtros centrado
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                "APLICAR FILTROS",
                                id="aplicar-filtros-btn",
                                className="login-button mt-2",
                                style={"width": "200px"}
                            ),
                        ], width={"size": 2, "offset": 5}),
                    ], className="mb-4"),
                ]
            ),
            
            # Botón de exportación PDF
            dbc.Row([
                dbc.Col([
                    export_button("defensive-pdf", "Informe de Rendimiento Defensivo"),
                ], width={"size": 4, "offset": 4}),
            ], className="mb-4"),
            
            # Contenido de la página
            html.Div(
                id="defensive-content",
                children=[
                    # Primera fila: Solo tabla de los mejores defensores ocupando todo el ancho
                    dbc.Row([
                        # Ranking de mejores defensores a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Top 10 Jugadores Defensivos", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="top-defensores-table")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Segunda fila: Gráfico de dispersión ocupando todo el ancho
                    dbc.Row([
                        # Gráfico de dispersión a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    # Selector de métricas para el gráfico
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(className="filter-title", children="Métrica X"),
                                            dcc.Dropdown(
                                                id="scatter-x-metric",
                                                options=[],  # Se llenará dinámicamente
                                                placeholder="Selecciona métrica para eje X",
                                                className="filter-dropdown mb-2",
                                                style={"color": "black"}
                                            ),
                                        ], width=5),
                                        dbc.Col([
                                            html.Div(className="filter-title", children="Métrica Y"),
                                            dcc.Dropdown(
                                                id="scatter-y-metric",
                                                options=[],  # Se llenará dinámicamente
                                                placeholder="Selecciona métrica para eje Y",
                                                className="filter-dropdown mb-2",
                                                style={"color": "black"}
                                            ),
                                        ], width=5),
                                        dbc.Col([
                                            html.Button(
                                                "ACTUALIZAR GRÁFICO",
                                                id="actualizar-grafico-btn",
                                                className="login-button mt-4",
                                                style={"width": "100%"}
                                            ),
                                        ], width=2),
                                    ]),
                                    html.Div(id="scatter-metricas-defensivas")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Tercera fila: Comparativa entre ligas (ancho completo)
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.Div(id="comparativa-ligas")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Cuarta fila: Distribución de métricas (ancho completo)
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.Div(id="distribucion-metricas")
                                ]
                            )
                        ], width=12),
                    ]),
                ]
            ),
            
            # Store para los datos filtrados
            dcc.Store(id="filtered-data-store"),
        ]
    )
])
            
        
# Callback para cargar opciones de filtros iniciales
@callback(
    [Output("posicion-filter", "options"),
     Output("liga-filter", "options")],
    Input("posicion-filter", "id")  # Trigger dummy
)
def load_filter_options(dummy):
    df = data_manager.get_data()
    
    # Obtener opciones de posición
    posiciones = []
    if not df.empty and 'POSICIÓN' in df.columns:
        posiciones = [{"label": pos, "value": pos} for pos in sorted(df['POSICIÓN'].unique())]
    
    # Obtener opciones de liga exclusivamente de la columna LIGA
    ligas = []
    if not df.empty and 'LIGA' in df.columns:
        # Asegurarse de eliminar valores nulos o vacíos
        ligas_unicas = [liga for liga in df['LIGA'].dropna().unique() if liga and str(liga).strip()]
        # Añadir la opción "Todas las ligas" al principio
        ligas = [{"label": "Todas las ligas", "value": "todas"}] + [{"label": liga, "value": liga} for liga in sorted(ligas_unicas)]
    
    return posiciones, ligas

# NUEVO: Callback para actualizar opciones de equipos basados en ligas seleccionadas
@callback(
    Output("equipo-filter", "options"),
    Input("liga-filter", "value")
)
def update_team_options(ligas_seleccionadas):
    # Si no hay ligas seleccionadas, devolver opción "Todos los equipos"
    if not ligas_seleccionadas:
        return [{"label": "Todos los equipos", "value": "todos"}]
    
    df = data_manager.get_data()
    
    # Verificar que tenemos la columna EQUIPO
    if 'EQUIPO' not in df.columns:
        # Buscar alternativas como CLUB si EQUIPO no está disponible
        equipo_col = None
        for col in ['CLUB', 'TEAM', 'EQUIPO']:
            if col in df.columns:
                equipo_col = col
                break
        
        if not equipo_col:
            return [{"label": "Datos de equipo no disponibles", "value": "no_data"}]
    else:
        equipo_col = 'EQUIPO'
    
    # Si "todas" está en las ligas seleccionadas, no filtrar por liga
    if "todas" in ligas_seleccionadas:
        equipos = [{"label": "Todos los equipos", "value": "todos"}]
        equipos_unicos = df[equipo_col].dropna().unique()
        equipos.extend([{"label": equipo, "value": equipo} for equipo in sorted(equipos_unicos)])
        return equipos
    
    # Filtrar el DataFrame para incluir solo las ligas seleccionadas
    filtered_df = df[df['LIGA'].isin(ligas_seleccionadas)]
    
    # Obtener equipos únicos de las ligas seleccionadas
    equipos = []
    if not filtered_df.empty:
        # Añadir opción para todos los equipos al principio
        equipos = [{"label": "Todos los equipos", "value": "todos"}]
        
        # Obtener equipos únicos de las ligas seleccionadas
        equipos_unicos = filtered_df[equipo_col].dropna().unique()
        equipos.extend([{"label": equipo, "value": equipo} for equipo in sorted(equipos_unicos)])
    
    return equipos

# Callback para aplicar filtros y actualizar datos
@callback(
    Output("filtered-data-store", "data"),
    [Input("aplicar-filtros-btn", "n_clicks")],
    [State("posicion-filter", "value"),
     State("liga-filter", "value"),
     State("equipo-filter", "value"),  # Filtro de equipos
     State("metricas-filter", "value"),  # Filtro de métricas
     State("edad-range-slider", "value"),
     State("minutos-slider", "value")],
    prevent_initial_call=False  # Cambia a False para permitir ejecución en carga
)
def filter_data(n_clicks, posiciones, ligas, equipos, metricas_seleccionadas, rango_edad, min_minutos):
    df = data_manager.get_data()
    
    if df.empty:
        return []
    
    # Si no se ha hecho clic (carga inicial), retornar datos de defensas con filtros por defecto
    if not n_clicks:
        # Para la carga inicial, aplicar solo filtros básicos por defecto
        filtered_df = df.copy()
        
        # Filtrar solo posiciones defensivas
        posiciones_defensivas = ["DEFENSA CENTRAL", "LATERAL DERECHO", "LATERAL IZQUIERDO", "CARRILERO DERECHO", 
                                "CARRILERO IZQUIERDO", "PIVOTE DEFENSIVO"]
        
        # Filtrar por posiciones defensivas (usar 'startswith' para capturar variaciones)
        filtered_df = filtered_df[filtered_df['POSICIÓN'].apply(
            lambda x: any(x.startswith(pos[:5]) for pos in posiciones_defensivas) if isinstance(x, str) else False
        )]
        
        # Filtrar por las principales ligas europeas
        if 'LIGA' in filtered_df.columns:
            ligas_principales = ["LALIGA", "SERIE A", "BUNDESLIGA", "PREMIER LEAGUE", "LIGUE 1"]
            
            # Crear una condición flexible para capturar variaciones en los nombres de las ligas
            liga_condition = filtered_df['LIGA'].apply(
                lambda x: any(liga.lower() in str(x).lower() for liga in ligas_principales) if x else False
            )
            
            # Aplicar el filtro de ligas
            filtered_df = filtered_df[liga_condition]
        
        # Si no hay defensas de estas ligas, volver al filtro original solo de defensas
        if filtered_df.empty:
            filtered_df = df.copy()
            filtered_df = filtered_df[filtered_df['POSICIÓN'].apply(
                lambda x: any(x.startswith(pos[:5]) for pos in posiciones_defensivas) if isinstance(x, str) else False
            )]
            
            # Si sigue vacío, usar el DataFrame completo
            if filtered_df.empty:
                filtered_df = df.copy()
        
        # Aplicar filtro de minutos jugados por defecto
        if 'MINUTOS JUGADOS' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['MINUTOS JUGADOS'] >= 500]  # Valor por defecto
        
        # Seleccionar columnas básicas siempre necesarias
        cols_to_include = ['NOMBRE', 'POSICIÓN']
        
        if 'EDAD' in filtered_df.columns:
            cols_to_include.append('EDAD')
        
        if 'MINUTOS JUGADOS' in filtered_df.columns:
            cols_to_include.append('MINUTOS JUGADOS')
            
        # Incluir columnas de equipo y liga si están disponibles
        if 'LIGA' in filtered_df.columns:
            cols_to_include.append('LIGA')
        
        for col in ['EQUIPO', 'CLUB', 'TEAM']:
            if col in filtered_df.columns:
                cols_to_include.append(col)
                break
        
        # Incluir métricas defensivas por defecto
        default_metrics = ["DUELOS TERRESTRES GANADOS", "INTERCEPCIONES", "RECUPERACIONES BALONES SUELTOS"]
        for metric in default_metrics:
            if metric in filtered_df.columns:
                cols_to_include.append(metric)
        
        # Filtrar DataFrame
        filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 3)
        
        return {
            'data': filtered_df.to_dict('records'),
            'selected_metrics': default_metrics
        }
    
    # Si se ha hecho clic, aplicar todos los filtros
    # Hacer una copia de los datos
    filtered_df = df.copy()
    
    # Aplicar filtro de posición
    if posiciones and len(posiciones) > 0:
        filtered_df = filtered_df[filtered_df['POSICIÓN'].isin(posiciones)]
    
    # Aplicar filtro de liga solo en la columna LIGA (excluyendo "todas")
    if ligas and len(ligas) > 0 and 'LIGA' in filtered_df.columns and 'todas' not in ligas:
        filtered_df = filtered_df[filtered_df['LIGA'].isin(ligas)]
    
    # Aplicar filtro de equipos si no está en "todos"
    if equipos and len(equipos) > 0 and 'todos' not in equipos:
        # Buscar columna de equipo disponible
        equipo_col = None
        for col in ['EQUIPO', 'CLUB', 'TEAM']:
            if col in filtered_df.columns:
                equipo_col = col
                break
        
        if equipo_col:
            filtered_df = filtered_df[filtered_df[equipo_col].isin(equipos)]
    
    # Aplicar filtro de edad
    if 'FECHA NACIMIENTO' in filtered_df.columns:
        min_edad, max_edad = rango_edad
        try:
            fecha_nacimiento = pd.to_datetime(filtered_df['FECHA NACIMIENTO'], errors='coerce')
            filtered_df['EDAD'] = pd.to_datetime('now').year - fecha_nacimiento.dt.year
            filtered_df = filtered_df[(filtered_df['EDAD'] >= min_edad) & (filtered_df['EDAD'] <= max_edad)]
        except Exception as e:
            print(f"Error al calcular edades: {e}")
    
    # Aplicar filtro de minutos jugados
    if 'MINUTOS JUGADOS' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['MINUTOS JUGADOS'] >= min_minutos]
    
    # Seleccionar columnas básicas siempre necesarias
    cols_to_include = ['NOMBRE', 'POSICIÓN', 'EDAD', 'MINUTOS JUGADOS']
    
    # Incluir columnas de equipo y liga si están disponibles
    if 'LIGA' in filtered_df.columns:
        cols_to_include.append('LIGA')
    
    for col in ['EQUIPO', 'CLUB', 'TEAM']:
        if col in filtered_df.columns:
            cols_to_include.append(col)
            break
    
    # Incluir solo las métricas defensivas seleccionadas
    if metricas_seleccionadas and len(metricas_seleccionadas) > 0:
        # Añadir solo las métricas seleccionadas que existen en el DataFrame
        for col in metricas_seleccionadas:
            if col in filtered_df.columns:
                cols_to_include.append(col)
    else:
        # Si no hay métricas seleccionadas, incluir todas las disponibles
        for col in metricas_defensivas:
            if col in filtered_df.columns:
                cols_to_include.append(col)
    
    # Seleccionar solo columnas disponibles
    cols_to_include = [col for col in cols_to_include if col in filtered_df.columns]
    
    # Seleccionar solo las columnas deseadas y filtrar filas con valores faltantes
    filtered_df = filtered_df[cols_to_include]
    filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 3)  # Permitir algunos valores faltantes
    
    # Guardar métricas seleccionadas en los datos (para usar en otros callbacks)
    selected_metrics_data = {
        'data': filtered_df.to_dict('records'),
        'selected_metrics': metricas_seleccionadas if metricas_seleccionadas else metricas_defensivas
    }
    
    return selected_metrics_data

# NUEVO: Callback para actualizar las opciones de métricas en los dropdowns del gráfico
@callback(
    [Output("scatter-x-metric", "options"),
     Output("scatter-y-metric", "options"),
     Output("scatter-x-metric", "value"),
     Output("scatter-y-metric", "value")],
    Input("metricas-filter", "value")
)
def update_scatter_metric_options(selected_metrics):
    # Si no hay métricas seleccionadas, mostrar mensaje de error
    if not selected_metrics or len(selected_metrics) == 0:
        options = [{"label": "Selecciona métricas en el filtro", "value": ""}]
        return options, options, None, None
    
    # Crear opciones solo para las métricas seleccionadas
    options = []
    for metric in selected_metrics:
        # Crear etiquetas más legibles para las métricas
        label = metric.replace("_", " ").title()
        options.append({"label": label, "value": metric})
    
    # Seleccionar por defecto las dos primeras métricas seleccionadas
    x_default = selected_metrics[0] if len(selected_metrics) > 0 else None
    y_default = selected_metrics[1] if len(selected_metrics) > 1 else selected_metrics[0]
    
    return options, options, x_default, y_default

# Callback para crear gráfico de dispersión actualizado con métricas seleccionadas
@callback(
    Output("scatter-metricas-defensivas", "children"),
    [Input("actualizar-grafico-btn", "n_clicks"),
     Input("filtered-data-store", "data")],
    [State("scatter-x-metric", "value"),
     State("scatter-y-metric", "value")]
)
def create_scatter_plot(n_clicks, filtered_data, x_metric, y_metric):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
    else:
        raw_data = filtered_data
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.",
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Verificar si se han proporcionado las métricas X e Y
    if not x_metric or not y_metric:
        return html.Div("Selecciona métricas en el filtro y haz clic en 'ACTUALIZAR GRÁFICO'", 
                       style={"text-align": "center", "padding": "20px", "font-weight": "bold", "color": "#00BFFF"})
    
    # Verificar métricas disponibles
    if x_metric in df.columns and y_metric in df.columns:
        # Crear formato humanizado para los títulos de los ejes
        x_title = x_metric.replace("_", " ").title()
        y_title = y_metric.replace("_", " ").title()
        
        # Crear gráfico de dispersión mejorado
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            hover_name='NOMBRE',
            size='MINUTOS JUGADOS',
            size_max=20,
            template="plotly_dark",
            labels={
                x_metric: x_title,
                y_metric: y_title,
                'POSICIÓN': 'Posición',
                'MINUTOS JUGADOS': 'Minutos Jugados'
            },
            opacity=0.9,
            custom_data=['NOMBRE', 'POSICIÓN']
        )
        # Después de crear el gráfico, añade estas líneas para modificar los marcadores:
        fig.update_traces(
            marker=dict(
                color='#00BFFF',       # Azul claro
                line=dict(width=2, color='#FFFF00')  # Borde amarillo
            )
        )
        
        # Diseño hover personalizado
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Posición: %{customdata[1]}<br>" +
                         f"{x_title}: %{{x:.2f}}<br>" +
                         f"{y_title}: %{{y:.2f}}<br>" +
                         "Minutos: %{marker.size:.0f}<extra></extra>"
        )
        
        # Mejorar el diseño del gráfico
        fig.update_layout(
            title={
                'text': f"{x_title} vs. {y_title} por Jugador",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22, 'color': 'white', 'family': 'Arial, sans-serif'}
            },
            font=dict(
                family="Arial, sans-serif",
                size=13,
                color="white"
            ),
            legend=dict(
                title="POSICIÓN",
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text=x_title, font=dict(size=16, color="white")),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.2)",
                zerolinewidth=1
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(size=16, color="white")),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.2)",
                zerolinewidth=1
            ),
            height=550,  # Mayor altura
            margin=dict(l=10, r=10, t=80, b=10)
        )
        
        # Agregar líneas de referencia de cuadrantes (valores medios)
        x_mean = df[x_metric].mean()
        y_mean = df[y_metric].mean()
        
        # Línea vertical en la media de x
        fig.add_shape(
            type="line",
            x0=x_mean,
            y0=df[y_metric].min(),
            x1=x_mean,
            y1=df[y_metric].max(),
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash")
        )
        
        # Línea horizontal en la media de y
        fig.add_shape(
            type="line",
            x0=df[x_metric].min(),
            y0=y_mean,
            x1=df[x_metric].max(),
            y1=y_mean,
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash")
        )
        
        # Añadir anotaciones para los cuadrantes
        fig.add_annotation(
            x=df[x_metric].max() - (df[x_metric].max() - df[x_metric].min())*0.1,
            y=df[y_metric].max() - (df[y_metric].max() - df[y_metric].min())*0.1,
            text="Rendimiento Superior",
            showarrow=False,
            font=dict(color="rgba(0, 255, 0, 0.7)", size=12)
        )
        
        fig.add_annotation(
            x=df[x_metric].min() + (df[x_metric].max() - df[x_metric].min())*0.1,
            y=df[y_metric].max() - (df[y_metric].max() - df[y_metric].min())*0.1,
            text="Alto " + y_title,
            showarrow=False,
            font=dict(color="rgba(255, 255, 0, 0.7)", size=12)
        )
        
        # Añadir una explicación del gráfico
        explanation = html.Div([
            html.H4("Análisis de dispersión", style={"color": "#00BFFF", "margin-bottom": "10px"}),
            html.P([
                "Este gráfico muestra la relación entre ", 
                html.Strong(x_title), 
                " y ", 
                html.Strong(y_title),
                " para cada jugador. El tamaño de los puntos representa los minutos jugados."
            ]),
            html.P([
                "Las líneas punteadas muestran los valores promedio, dividiendo el gráfico en cuadrantes.",
                " Los jugadores en el cuadrante superior derecho destacan en ambas métricas."
            ]),
        ], style={"margin-top": "20px", "padding": "15px", "background-color": "rgba(0,0,0,0.2)", "border-radius": "5px"})
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False}),
            explanation
        ])
    else:
        return html.Div(f"Las métricas seleccionadas no están disponibles en los datos filtrados.",
                       style={"text-align": "center", "padding": "20px", "color": "orange"})

# Callback para crear tabla de top defensores
# Callback para crear tabla de top defensores con fotos
@callback(
    Output("top-defensores-table", "children"),
    Input("filtered-data-store", "data")
)
def create_top_defensores_table(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.")
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_defensivas
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.")
    
    # Crear un score defensivo basado en las métricas seleccionadas
    defense_score = pd.Series(0, index=df.index)
    metrics_used = []
    
    for metrica in selected_metrics:
        if metrica in df.columns:
            metrics_used.append(metrica)
            # Normalizar la métrica (0-1) y añadirla al score
            if df[metrica].max() > df[metrica].min():
                normalized = (df[metrica] - df[metrica].min()) / (df[metrica].max() - df[metrica].min())
                defense_score += normalized
    
    if not metrics_used:
        return html.Div("No se encontraron métricas seleccionadas en los datos disponibles.")
    
    df['Defense_Score'] = defense_score
    
    # Ordenar por score defensivo y tomar los 10 mejores
    top_10 = df.sort_values('Defense_Score', ascending=False).head(10)
    
    # Seleccionar columnas para mostrar, añadiendo FOTO al principio
    display_cols = ['FOTO', 'NOMBRE', 'POSICIÓN']
    
    # Columnas de datos (excluir FOTO que es virtual)
    data_cols = ['NOMBRE', 'POSICIÓN']
    
    # Añadir EQUIPO si está disponible
    for col in ['EQUIPO', 'CLUB', 'TEAM']:
        if col in top_10.columns:
            display_cols.append(col)
            data_cols.append(col)
            break
    
    # Añadir LIGA si está disponible
    if 'LIGA' in top_10.columns:
        display_cols.append('LIGA')
        data_cols.append('LIGA')
    
    # Añadir EDAD si está disponible
    if 'EDAD' in top_10.columns:
        display_cols.append('EDAD')
        data_cols.append('EDAD')
    
    # Añadir las métricas seleccionadas disponibles
    metrics_to_show = [m for m in selected_metrics if m in top_10.columns]
    display_cols.extend(metrics_to_show)
    data_cols.extend(metrics_to_show)
    
    # Crear los encabezados de la tabla
    header_cells = []
    
    # Añadir encabezado vacío para la columna de fotos
    header_cells.append(
        html.Th("", style={
            'backgroundColor': '#00BFFF',
            'color': 'white',
            'padding': '10px',
            'textAlign': 'center',
            'fontWeight': 'bold',
            'width': '60px'
        })
    )
    
    # Añadir resto de encabezados
    for col in data_cols:
        header_cells.append(
            html.Th(col, style={
                'backgroundColor': '#00BFFF',
                'color': 'white',
                'padding': '10px',
                'textAlign': 'center',
                'fontWeight': 'bold'
            })
        )
    
    header_row = html.Tr(header_cells)
    
    # Crear las filas de la tabla
    table_rows = []
    for i, (_, row) in enumerate(top_10.iterrows()):
        # Alternar colores de fondo
        bg_color = '#0E2B3D' if i % 2 == 0 else '#1A1A1A'
        row_cells = []
        
        # Añadir celda con foto del jugador
        player_name = row['NOMBRE']
        imagen_url = data_manager.get_player_image_url(player_name)
        
        row_cells.append(
            html.Td(
                html.Img(
                    src=imagen_url if imagen_url else "/assets/players/placeholder.png",
                    style={
                        "width": "40px",
                        "height": "40px",
                        "border-radius": "50%",
                        "object-fit": "cover",
                        "border": "2px solid #00BFFF",
                    }
                ),
                style={"padding": "5px", "text-align": "center", "width": "60px"}
            )
        )
        
        # Añadir el resto de celdas con datos
        for col in data_cols:
            value = row[col]
            # Formatear valores correctamente
            if isinstance(value, (int, float)) and not pd.isna(value):
                if col in metrics_to_show:
                    formatted = f"{value:.2f}"
                elif col == 'EDAD':
                    formatted = f"{int(value)}" if not pd.isna(value) else "-"
                else:
                    formatted = f"{value:.1f}" if value % 1 != 0 else f"{int(value)}"
            else:
                formatted = str(value) if not pd.isna(value) else "-"
            
            # Crear celda
            row_cells.append(html.Td(formatted, style={'padding': '10px'}))
        
        # Añadir fila a la tabla
        table_rows.append(html.Tr(row_cells, style={'backgroundColor': bg_color}))
    
    # Crear tabla completa
    table = html.Table(
        [html.Thead(header_row), html.Tbody(table_rows)],
        style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'color': 'white'
        }
    )
    
    return table

# Callback para crear comparativa entre ligas
@callback(
    Output("comparativa-ligas", "children"),
    Input("filtered-data-store", "data")
)
def create_league_comparison(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_defensivas
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    # Verificar si existe la columna LIGA
    if 'LIGA' not in df.columns:
        return html.Div("No hay datos de liga disponibles para comparar.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Limitar ligas para mejor visualización (top 5 por número de jugadores)
    liga_counts = df['LIGA'].value_counts().head(5)
    df_filtered = df[df['LIGA'].isin(liga_counts.index)]
    
    # Seleccionar métricas disponibles de entre las seleccionadas
    metrics = [m for m in selected_metrics if m in df.columns]
    
    # Limitar a 3 métricas para mejor visualización
    metrics = metrics[:3]
    
    if not metrics:
        return html.Div("No hay métricas defensivas disponibles para comparar.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Formatear nombres de métricas para mostrar
    formatted_metrics = [m.replace("_", " ").title() for m in metrics]
    
    # Agrupar por liga y calcular la media de las métricas
    league_stats = df_filtered.groupby('LIGA')[metrics].mean().reset_index()
    
    # Definir colores personalizados
    custom_colors = ['#00BFFF', '#4CAF50', '#FFFF00']
    
    # Crear gráfico de barras para comparar ligas
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        # Obtener el valor formateado de la métrica para mostrar
        formatted_metric = formatted_metrics[i]
        
        fig.add_trace(go.Bar(
            x=league_stats['LIGA'],
            y=league_stats[metric],
            name=formatted_metric,
            marker_color=custom_colors[i % len(custom_colors)],
            hovertemplate='<b>%{x}</b><br>' +
                         f'{formatted_metric}: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='group',
        title={
            'text': "Comparativa de Métricas Defensivas por Liga",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Liga", font=dict(size=14, color="white")),
            tickangle=30,
            tickfont=dict(size=12),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text="Valor Promedio", font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            font=dict(size=12)
        ),
        height=450,
        margin=dict(l=10, r=10, t=80, b=50)
    )
    
    # Añadir un título y explicación
    return html.Div([
        html.H3("Comparativa por Liga", style={"color": "#00BFFF", "margin-bottom": "15px"}),
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.P("Este gráfico muestra el rendimiento promedio por liga para las métricas seleccionadas.",
                 style={"font-style": "italic", "margin-top": "10px"})
        ])
    ])

# Callback para crear distribución de métricas defensivas
@callback(
    Output("distribucion-metricas", "children"),
    Input("filtered-data-store", "data")
)
def create_metrics_distribution(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_defensivas
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    # Seleccionar la primera métrica disponible de las seleccionadas
    metric = None
    for m in selected_metrics:
        if m in df.columns:
            metric = m
            break
    
    # Si no encontramos ninguna métrica en las seleccionadas, buscar entre todas
    if not metric:
        for m in metricas_defensivas:
            if m in df.columns:
                metric = m
                break
    
    if not metric:
        return html.Div("No hay métricas defensivas disponibles para mostrar la distribución.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Formatear nombre de la métrica para mostrar
    metric_title = metric.replace("_", " ").title()
    
    # Crear histograma mejorado
    fig = px.histogram(
        df,
        x=metric,
        title=f"Distribución de {metric_title}",
        template="plotly_dark",
        marginal="box",
        opacity=0.9,
        labels={metric: metric_title, 'POSICIÓN': 'Posición'},
        color_discrete_map={'DEFENSA_CENTRAL': '#00BFFF'}  # Asignar azul claro
    )

    # Después de crear el histograma, añade estas líneas:
    fig.update_traces(
        marker=dict(
            color='#00BFFF',       # Azul claro
            line=dict(width=1, color='#FFFF00')  # Borde amarillo
        )
    )

    
    # Añadir línea de valor medio
    mean_value = df[metric].mean()
    
    fig.add_shape(
        type="line",
        x0=mean_value,
        x1=mean_value,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=mean_value,
        y=0.95,
        yref="paper",
        text=f"Media: {mean_value:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-30,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="red",
        borderwidth=1
    )
    
    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text=metric_title, font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        yaxis=dict(
            title=dict(text="Número de Jugadores", font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        legend=dict(
            title="POSICIÓN",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            font=dict(size=11)
        ),
        height=450,
        margin=dict(l=10, r=10, t=50, b=50)
    )
    
    # Añadir un título y explicación
    return html.Div([
        html.H3("Distribución de Métricas", style={"color": "#00BFFF", "margin-bottom": "15px"}),
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.P([
                "Este histograma muestra la distribución de ",
                html.Strong(metric_title),
                " entre los jugadores. La línea roja representa el valor promedio."
            ], style={"font-style": "italic", "margin-top": "10px"})
        ])
    ])
    
    # Callback para abrir el modal
@callback(
    Output("defensive-pdf-modal", "is_open"),
    [Input("defensive-pdf-button", "n_clicks"),
     Input("defensive-pdf-modal-close", "n_clicks"),
     Input("defensive-pdf-modal-generate", "n_clicks")],
    [State("defensive-pdf-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n_open, n_close, n_generate, is_open):
    if n_open or n_close or n_generate:
        return not is_open
    return is_open

@callback(
    Output("defensive-pdf-download", "data"),
    Input("defensive-pdf-modal-generate", "n_clicks"),
    [State("defensive-pdf-title", "value"),
     State("defensive-pdf-description", "value"),
     State("filtered-data-store", "data")],
    prevent_initial_call=True
)
def generate_pdf(n_clicks, title, description, filtered_data):
    if not n_clicks:
        raise PreventUpdate
    
    import base64
    import io
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    # Crear un buffer para guardar el PDF
    buffer = io.BytesIO()
    
    # Crear el documento PDF con orientación horizontal
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), 
                           leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    
    # Lista para los elementos del PDF
    elements = []
    
    # Estilos
    styles = getSampleStyleSheet()
    
    # Crear estilo personalizado para el título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        textColor=colors.cyan,
        alignment=1,  # Centrado
        fontSize=28,
        fontName='Helvetica-Bold',
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        textColor=colors.white,
        fontSize=14,
        fontName='Helvetica-Bold',
        alignment=1  # Centrado
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        textColor=colors.white,
        fontSize=12,
        fontName='Helvetica'
    )
    
    # Agregar título
    elements.append(Paragraph(title or "Informe de Rendimiento Defensivo", title_style))
    elements.append(Spacer(1, 15))
    
    # Agregar fecha
    from datetime import datetime
    elements.append(Paragraph(f"Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # Agregar descripción si existe
    if description:
        elements.append(Paragraph(description, normal_style))
        elements.append(Spacer(1, 10))
    
    # Extraer datos
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        df = pd.DataFrame(filtered_data['data'])
        metrics = filtered_data.get('selected_metrics', [])
    else:
        df = pd.DataFrame()
        metrics = []
    
    # Agregar filtros aplicados
    if metrics:
        metrics_text = f"Filtros aplicados: Métricas: {', '.join(metrics[:3])}"
        if len(metrics) > 3:
            metrics_text += "..."
        metrics_text += f" | Jugadores analizados: {len(df)}"
        
        elements.append(Paragraph(metrics_text, ParagraphStyle(
            'CustomFilter',
            parent=styles['Normal'],
            textColor=colors.yellow,
            fontSize=11,
            fontName='Helvetica',
            alignment=1  # Centrado
        )))
        elements.append(Spacer(1, 15))
    
    # Crear tabla de jugadores
    if not df.empty:
        # Ordenar por alguna métrica relevante
        sort_metric = None
        for metric in ["DUELOS TERRESTRES GANADOS", "INTERCEPCIONES", "RECUPERACIONES BALONES SUELTOS"]:
            if metric in df.columns:
                sort_metric = metric
                break
        
        if sort_metric:
            df = df.sort_values(sort_metric, ascending=False)
        
        # Seleccionar columnas de interés
        headers = ["Nombre", "Posición", "Liga"]
        metrics_to_show = []
        
        # Añadir métricas principales - ABREVIAR los nombres
        for metric in metrics[:5]:
            if metric in df.columns:
                # Abreviar los nombres de las métricas para que quepan mejor
                short_name = metric
                short_name = short_name.replace("DUELOS TERRESTRES", "Duelos")
                short_name = short_name.replace("RECUPERACIONES BALONES SUELTOS", "Recup. BAL.")
                short_name = short_name.replace("INTERCEPCIONES", "INTERCEP.")
                short_name = short_name.replace("GANADOS", "GAN.")
                short_name = short_name.replace("PERDIDOS", "PERD.")
                short_name = short_name.replace("DEFENSIVOS", "DEF.")
                short_name = short_name.replace("PRIMER TERCIO", "P.T.")
                
                # Limitar longitud a 12 caracteres para evitar desbordes
                if len(short_name) > 12:
                    short_name = short_name[:12]
                
                headers.append(short_name)
                metrics_to_show.append(metric)
        
        # Añadir métricas adicionales si hay espacio
        if "BLOQUEOS" in df.columns and "BLOQUEOS" not in metrics_to_show:
            headers.append("BLOQUEOS")
            metrics_to_show.append("BLOQUEOS")
        
        if "DESPEJES CABEZA" in df.columns and "DESPEJES CABEZA" not in metrics_to_show:
            headers.append("DESP.CAB.")
            metrics_to_show.append("DESPEJES CABEZA")
        
        if "TOQUES DEFENSIVOS PRIMER TERCIO" in df.columns and "TOQUES DEFENSIVOS PRIMER TERCIO" not in metrics_to_show:
            headers.append("TOQ.DEF.PT")
            metrics_to_show.append("TOQUES DEFENSIVOS PRIMER TERCIO")
        
        # Preparar datos para la tabla
        table_data = [headers]
        
        # Añadir filas de datos - Acortar nombres si son muy largos
        for _, row in df.head(10).iterrows():
            nombre = row["NOMBRE"]
            if len(nombre) > 18:  # Limitar largo de nombres
                nombre_parts = nombre.split()
                if len(nombre_parts) > 1:
                    nombre = f"{nombre_parts[0]} {nombre_parts[-1]}"
                else:
                    nombre = nombre[:18]
                    
            row_data = [
                nombre,
                row["POSICIÓN"],
                row.get("LIGA", "")
            ]
            
            # Añadir métricas
            for metric in metrics_to_show:
                value = row.get(metric, "")
                if isinstance(value, (int, float)):
                    row_data.append(f"{value:.2f}")
                else:
                    row_data.append(str(value))
            
            table_data.append(row_data)
        
        # AJUSTAR TAMAÑOS DE COLUMNAS - Más espacio para métricas
        # Calcular anchos proporcionales al espacio disponible (750 puntos disponibles aprox.)
        page_width = 750
        num_cols = len(headers)
        
        # Distribuir el ancho disponible
        col_widths = []
        col_widths.append(110)  # Nombre
        col_widths.append(85)   # Posición
        col_widths.append(65)   # Liga
        
        # Calcular espacio restante para métricas
        metrics_space = page_width - sum(col_widths) - 10
        metrics_width = metrics_space / len(metrics_to_show)
        
        # Añadir anchos para métricas
        for _ in metrics_to_show:
            col_widths.append(metrics_width)
        
        # Crear tabla con los anchos calculados
        table = Table(table_data, colWidths=col_widths)
        
        # Estilo de tabla
        table_style = [
            # Cabecera
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Cuerpo de tabla
            ('BACKGROUND', (0, 1), (-1, -1), colors.navy),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 1), (2, -1), 'LEFT'),      # Alineación izquierda para nombre, posición, liga
            ('ALIGN', (3, 1), (-1, -1), 'CENTER'),   # Alineación central para métricas
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),       # Texto más pequeño para que quepa
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
        ]
        
        # Alternar colores de filas
        for i in range(1, len(table_data), 2):
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.darkblue))
        
        table.setStyle(TableStyle(table_style))
        elements.append(table)
    else:
        elements.append(Paragraph("No hay datos disponibles", normal_style))
    
    # Agregar pie de página
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("SOCCER DATA DBC - Análisis de Rendimiento Deportivo", 
                             ParagraphStyle('Footer', parent=styles['Normal'],
                                           textColor=colors.cyan, alignment=1, fontSize=10)))
    
    # Crear una clase para manejar el fondo negro
    class BlackBackground(SimpleDocTemplate):
        def __init__(self, *args, **kwargs):
            SimpleDocTemplate.__init__(self, *args, **kwargs)
        
        def beforePage(self):
            self.canv.setFillColor(colors.black)
            self.canv.rect(0, 0, self.width + 60, self.height + 60, fill=1)
    
    # Usar nuestra clase personalizada
    doc = BlackBackground(buffer, pagesize=landscape(letter),
                         leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    
    # Construir el PDF
    doc.build(elements)
    
    # Obtener los datos del buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Codificar en base64 para enviarlo
    pdf_base64 = base64.b64encode(pdf_data).decode('ascii')
    
    # Devolver el PDF para descarga
    return {
        "content": pdf_base64,
        "filename": f"informe_defensivo_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "type": "application/pdf",
        "base64": True
    }