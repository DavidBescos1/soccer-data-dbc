from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import json

from components.data_manager import DataManager
from components.sidebar import create_sidebar

# Inicializar el gestor de datos
data_manager = DataManager()

# Obtener la lista de ligas disponibles (para los logos)
ligas = [
    "allsvenskan", "seriea", "laliga", "premier", "ligue1", "Bundesliga",
    "eredivisie", "proleague", "danish", "eliteserien", "trendyol", 
    "russian", "brasileirao", "profesional", "chile", "Uruguay",
    "portugal", "saudi" 
]

def create_league_logos():
    """Crea el contenedor con los logos de las ligas de manera ordenada"""
    logos_container = html.Div(
        className="leagues-container",
        children=[
            html.Img(
                src=f"/assets/logos_ligas/{liga}.png",
                className="league-logo",
                title=liga.capitalize(),
                style={
                    "width": "40px", 
                    "height": "40px", 
                    "margin": "5px",
                    "object-fit": "contain",
                    "filter": "brightness(1.2)",
                    "transition": "transform 0.3s ease"
                }
            ) for liga in ligas
        ],
        style={
            "display": "flex",
            "flex-wrap": "wrap",
            "justify-content": "center",
            "align-items": "center",
            "gap": "10px",
            "margin-bottom": "20px",
            "padding": "10px",
            "background-color": "#1A1A1A",
            "border-radius": "10px",
            "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.2)"
        }
    )
    return logos_container

def get_dashboard_data():
    """
    Procesa los datos para el dashboard y asegura tipos serializables a JSON
    """
    df = data_manager.get_data()
    
    if df.empty:
        return {
            "total_jugadores": 0,
            "posiciones": [],
            "ligas": 0,
            "edades": [],
            "nacionalidades": [],
        }
    
    # Obtener estadísticas
    total_jugadores = int(df['NOMBRE'].nunique())
    
    # Posiciones - Asegurar que sean reales (no más de las que existen)
    posiciones = df['POSICIÓN'].value_counts().reset_index()
    posiciones.columns = ['Posición', 'Cantidad']
    # Asegurar un número realista de posiciones
    posiciones_list = posiciones.to_dict('records')
    
    # Lugares de nacimiento (no países) - excluir "SIN DATOS"
    lugares_count = 0
    nacionalidades_list = []
    if 'LUGAR NACIMIENTO' in df.columns:
        # Filtrar para excluir "SIN DATOS" y valores nulos
        lugares_filtrados = df[~df['LUGAR NACIMIENTO'].isin(["SIN DATOS", "", "UNKNOWN"]) & 
                              df['LUGAR NACIMIENTO'].notna()]
        
        lugares_count = lugares_filtrados['LUGAR NACIMIENTO'].nunique()
        
        # Obtener los lugares más comunes (top 10)
        nacionalidades = lugares_filtrados['LUGAR NACIMIENTO'].value_counts().reset_index()
        nacionalidades.columns = ['Lugar', 'Cantidad']
        nacionalidades = nacionalidades.head(10)  # Top 10 lugares
        
        # Convertir a lista de diccionarios
        nacionalidades_list = []
        for _, row in nacionalidades.iterrows():
            nacionalidades_list.append({
                'Lugar': str(row['Lugar']),
                'Cantidad': int(row['Cantidad'])
            })
    
    # Calcular edades a partir de la fecha de nacimiento
    edades_list = []
    if 'FECHA NACIMIENTO' in df.columns:
        # Inicializar columna EDAD
        df['EDAD'] = None
        try:
            # Convertir fechas con manejo de errores
            fecha_nacimiento = pd.to_datetime(df['FECHA NACIMIENTO'], errors='coerce')
            # Calcular edad solo para fechas válidas
            df.loc[fecha_nacimiento.notna(), 'EDAD'] = pd.to_datetime('now').year - fecha_nacimiento.dt.year
            # Agrupar por edad para estadísticas
            edades = df.dropna(subset=['EDAD']).groupby('EDAD').size().reset_index()
            edades.columns = ['Edad', 'Cantidad']
            # Filtrar edades en un rango razonable (16-45 años)
            edades = edades[(edades['Edad'] >= 16) & (edades['Edad'] <= 45)]
            
            # Convertir a lista de diccionarios
            edades_list = []
            for _, row in edades.iterrows():
                edades_list.append({
                    'Edad': int(row['Edad']),
                    'Cantidad': int(row['Cantidad'])
                })
        except Exception as e:
            print(f"Error al calcular edades: {e}")
    
    # Retornar datos con tipos Python nativos para asegurar serialización JSON
    return {
        "total_jugadores": total_jugadores,
        "posiciones": posiciones_list,
        "ligas": lugares_count,
        "edades": edades_list,
        "nacionalidades": nacionalidades_list,
    }

def create_position_chart(posiciones):
    """Crea el gráfico de distribución por posiciones"""
    # Verificar si posiciones es una lista de diccionarios y convertirla a DataFrame
    if isinstance(posiciones, list) and len(posiciones) > 0:
        posiciones = pd.DataFrame(posiciones)
    
    if isinstance(posiciones, pd.DataFrame) and posiciones.empty:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="No hay datos disponibles",
                showarrow=False,
                font=dict(color="white", size=16)
            )
        )
    
    # Crear gráfico de pie con mejor visualización
    fig = px.pie(
        posiciones, 
        values='Cantidad', 
        names='Posición',
        title='Distribución por Posiciones',
        color_discrete_sequence=px.colors.sequential.Plasma,
        hole=0.4
    )
    
    # Mejorar la visualización
    fig.update_traces(
        textposition='inside',
        textinfo='percent',
        textfont=dict(size=14, color='white'),
        pull=[0.05] * len(posiciones)
    )
    
    # Configurar la leyenda a la derecha
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.1,
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.2)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='#00BFFF', size=18),
        margin=dict(l=20, r=80, t=40, b=20),  # Más margen a la derecha para la leyenda
        height=400
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_age_chart(edades):
    """Crea el gráfico de distribución por edades"""
    # Verificar si edades es una lista de diccionarios y convertirla a DataFrame
    if isinstance(edades, list) and len(edades) > 0:
        edades = pd.DataFrame(edades)
        
    if isinstance(edades, pd.DataFrame) and edades.empty:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="No hay datos de edad disponibles",
                showarrow=False,
                font=dict(color="white", size=16)
            )
        )
    
    # Ordenar por edad
    edades = edades.sort_values('Edad')
    
    fig = px.bar(
        edades, 
        x='Edad', 
        y='Cantidad',
        title='Distribución por Edades',
        color_discrete_sequence=['#00BFFF']
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='#00BFFF', size=18),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=dict(text='Edad', font=dict(color='white')),
            gridcolor='#333',
            showgrid=True,
            tickmode='linear',
            tick0=16,
            dtick=5,
            range=[16, 45]
        ),
        yaxis=dict(
            title=dict(text='Cantidad de Jugadores', font=dict(color='white')),
            gridcolor='#333',
            showgrid=True
        ),
        height=350
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_nationality_chart(nacionalidades):
    """Crea el gráfico de lugares de nacimiento más comunes"""
    # Verificar si nacionalidades es una lista de diccionarios y convertirla a DataFrame
    if isinstance(nacionalidades, list) and len(nacionalidades) > 0:
        nacionalidades = pd.DataFrame(nacionalidades)
        
    if isinstance(nacionalidades, pd.DataFrame) and nacionalidades.empty:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="No hay datos de lugar de nacimiento disponibles",
                showarrow=False,
                font=dict(color="white", size=16)
            )
        )
    
    # Ordenar por cantidad descendente
    nacionalidades = nacionalidades.sort_values('Cantidad', ascending=False)
    
    # Limitar a top 10 lugares y renombrar la columna
    nacionalidades.columns = ['Lugar de Nacimiento', 'Cantidad'] if 'Lugar' in nacionalidades.columns else nacionalidades.columns
    
    fig = px.bar(
        nacionalidades, 
        y='Lugar de Nacimiento', 
        x='Cantidad',
        title='Lugares de Nacimiento Más Comunes',
        orientation='h',
        color='Cantidad',
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='#00BFFF', size=18),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=dict(text='Cantidad de Jugadores', font=dict(color='white')),
            gridcolor='#333',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text='Lugar de Nacimiento', font=dict(color='white')),
            gridcolor='#333',
            showgrid=False,
            categoryorder='total ascending'
        ),
        coloraxis_showscale=False,
        height=350
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_minutes_chart(df):
    """Crea un gráfico adicional de distribución de minutos jugados"""
    if df.empty or 'MINUTOS JUGADOS' not in df.columns:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="No hay datos de minutos jugados disponibles",
                showarrow=False,
                font=dict(color="white", size=16)
            )
        )
    
    # Crear bins para los minutos jugados
    df_filtered = df[df['MINUTOS JUGADOS'].notna()]
    
    # Crear bins de 500 minutos
    max_minutos = max(3000, df_filtered['MINUTOS JUGADOS'].max())
    bins = list(range(0, int(max_minutos) + 501, 500))
    
    # Crear etiquetas para los bins
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    
    # Crear columna con categoría de minutos jugados
    df_filtered['Rango de Minutos'] = pd.cut(df_filtered['MINUTOS JUGADOS'], bins=bins, labels=labels, right=False)
    
    # Contar jugadores por categoría
    minutos_counts = df_filtered['Rango de Minutos'].value_counts().reset_index()
    minutos_counts.columns = ['Rango de Minutos', 'Cantidad']
    minutos_counts = minutos_counts.sort_values('Rango de Minutos')
    
    # Crear gráfico
    fig = px.bar(
        minutos_counts, 
        x='Rango de Minutos', 
        y='Cantidad',
        title='Distribución por Minutos Jugados',
        color='Cantidad',
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='#00BFFF', size=18),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=dict(text='Rango de Minutos', font=dict(color='white')),
            gridcolor='#333',
            showgrid=True,
            tickangle=45  # Inclinar las etiquetas para mejor lectura
        ),
        yaxis=dict(
            title=dict(text='Cantidad de Jugadores', font=dict(color='white')),
            gridcolor='#333',
            showgrid=True
        ),
        coloraxis_showscale=False,
        height=350
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_posicion_minutos_chart(df):
    """Crea un gráfico que relaciona posición y minutos jugados promedio"""
    if df.empty or 'MINUTOS JUGADOS' not in df.columns or 'POSICIÓN' not in df.columns:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="No hay datos suficientes disponibles",
                showarrow=False,
                font=dict(color="white", size=16)
            )
        )
    
    # Filtrar datos válidos
    df_filtered = df[df['MINUTOS JUGADOS'].notna() & df['POSICIÓN'].notna()]
    
    # Calcular promedio de minutos por posición
    minutos_por_posicion = df_filtered.groupby('POSICIÓN')['MINUTOS JUGADOS'].mean().reset_index()
    minutos_por_posicion.columns = ['Posición', 'Promedio de Minutos']
    minutos_por_posicion = minutos_por_posicion.sort_values('Promedio de Minutos', ascending=False)
    
    # Crear gráfico de barras
    fig = px.bar(
        minutos_por_posicion, 
        x='Posición', 
        y='Promedio de Minutos',
        title='Promedio de Minutos Jugados por Posición',
        color='Promedio de Minutos',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='#00BFFF', size=18),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=dict(text='Posición', font=dict(color='white')),
            gridcolor='#333',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text='Promedio de Minutos', font=dict(color='white')),
            gridcolor='#333',
            showgrid=True
        ),
        coloraxis_showscale=False,
        height=350
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# Layout de la página de dashboard
layout = html.Div([
    # Sidebar
    create_sidebar(),
    
    # Contenido principal
    html.Div(
        className="content-container fade-in",
        children=[
            # Logos de ligas en la parte superior
            create_league_logos(),
            
            # Tarjetas resumen
            dbc.Row([
                dbc.Col([
                    html.Div(
                        className="dashboard-card",
                        children=[
                            html.Div(className="dashboard-card-title", children="Total Jugadores"),
                            html.Div(
                                id="total-jugadores-value",
                                className="dashboard-card-value"
                            )
                        ],
                        style={
                            "display": "flex",
                            "flex-direction": "column",
                            "align-items": "center",
                            "justify-content": "center",
                            "padding": "20px"
                        }
                    )
                ], width=4),
                
                dbc.Col([
                    html.Div(
                        className="dashboard-card",
                        children=[
                            html.Div(className="dashboard-card-title", children="Posiciones Registradas"),
                            html.Div(
                                id="posiciones-value",
                                className="dashboard-card-value"
                            )
                        ],
                        style={
                            "display": "flex",
                            "flex-direction": "column",
                            "align-items": "center",
                            "justify-content": "center",
                            "padding": "20px"
                        }
                    )
                ], width=4),
                
                dbc.Col([
                    html.Div(
                        className="dashboard-card",
                        children=[
                            html.Div(className="dashboard-card-title", children="Lugares de Nacimiento"),
                            html.Div(
                                id="paises-value",
                                className="dashboard-card-value"
                            )
                        ],
                        style={
                            "display": "flex",
                            "flex-direction": "column",
                            "align-items": "center",
                            "justify-content": "center",
                            "padding": "20px"
                        }
                    )
                ], width=4),
            ], className="mb-4"),
            
            # Primera fila: Gráficos de edad y nacionalidad
            dbc.Row([
                dbc.Col([
                    html.Div(
                        className="graph-container",
                        children=[
                            html.Div(id="age-chart")
                        ]
                    )
                ], width=6),
                
                dbc.Col([
                    html.Div(
                        className="graph-container",
                        children=[
                            html.Div(id="nationality-chart")
                        ]
                    )
                ], width=6),
            ], className="mb-4"),
            
            # Segunda fila: Gráficos adicionales
            dbc.Row([
                dbc.Col([
                    html.Div(
                        className="graph-container",
                        children=[
                            html.Div(id="minutes-chart")
                        ]
                    )
                ], width=6),
                
                dbc.Col([
                    html.Div(
                        className="graph-container",
                        children=[
                            html.Div(id="position-minutes-chart")
                        ]
                    )
                ], width=6),
            ], className="mb-4"),
            
            # Tercera fila: Gráfico de posiciones a ancho completo
            dbc.Row([
                dbc.Col([
                    html.Div(
                        className="graph-container",
                        children=[
                            html.Div(id="position-chart")
                        ]
                    )
                ], width=12),
            ]),
        ]
    ),
    
    # Store component para almacenar datos procesados
    dcc.Store(id='dashboard-data-store'),
])

# Callback para procesar y almacenar datos del dashboard
@callback(
    Output('dashboard-data-store', 'data'),
    Input('dashboard-data-store', 'id')
)
def process_dashboard_data(id):
    data = get_dashboard_data()
    try:
        # Verificar la serializabilidad antes de retornar
        json.dumps(data)
        return data
    except TypeError as e:
        print(f"Error: Los datos no son serializables a JSON: {e}")
        # Retornar un diccionario vacío que sea serializable
        return {
            "total_jugadores": 0,
            "posiciones": [],
            "ligas": 0,
            "edades": [],
            "nacionalidades": [],
        }

# Callback para actualizar las tarjetas resumen
@callback(
    [Output('total-jugadores-value', 'children'),
     Output('posiciones-value', 'children'),
     Output('paises-value', 'children')],
    Input('dashboard-data-store', 'data')
)
def update_summary_cards(data):
    if not data:
        return "0", "0", "0"
    
    total_jugadores = data.get("total_jugadores", 0)
    posiciones_count = len(data.get("posiciones", [])) 
    lugares_count = data.get("ligas", 0)
    
    # Quitar las comas en los números
    return f"{total_jugadores}", f"{posiciones_count}", f"{lugares_count}"

# Callback para actualizar los gráficos
@callback(
    [Output('position-chart', 'children'),
     Output('age-chart', 'children'),
     Output('nationality-chart', 'children'),
     Output('minutes-chart', 'children'),
     Output('position-minutes-chart', 'children')],
    Input('dashboard-data-store', 'data')
)
def update_charts(data):
    # Obtener datos originales para los nuevos gráficos
    df = data_manager.get_data()
    
    if not data or df.empty:
        position_chart = create_position_chart([])
        age_chart = create_age_chart([])
        nationality_chart = create_nationality_chart([])
        minutes_chart = create_minutes_chart(pd.DataFrame())
        position_minutes_chart = create_posicion_minutos_chart(pd.DataFrame())
    else:
        position_chart = create_position_chart(data.get("posiciones", []))
        age_chart = create_age_chart(data.get("edades", []))
        nationality_chart = create_nationality_chart(data.get("nacionalidades", []))
        minutes_chart = create_minutes_chart(df)
        position_minutes_chart = create_posicion_minutos_chart(df)
    
    return position_chart, age_chart, nationality_chart, minutes_chart, position_minutes_chart
