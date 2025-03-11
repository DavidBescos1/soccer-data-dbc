import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from flask_login import logout_user, current_user
from flask import redirect
from app import app, server
import os

# Importar todas las páginas
from pages import login, dashboard, jugadores_similares, rendimiento_defensivo, rendimiento_ofensivo, creacion_juego, analisis_superacion, nutricion_jugadores

# Diccionario de páginas disponibles
pages = {
    '/login': login.layout,
    '/dashboard': dashboard.layout,
    '/jugadores-similares': jugadores_similares.layout,
    '/rendimiento-defensivo': rendimiento_defensivo.layout,
    '/rendimiento-ofensivo': rendimiento_ofensivo.layout,
    '/creacion-juego': creacion_juego.layout,
    '/analisis-superacion': analisis_superacion.layout,
    '/nutricion-jugadores': nutricion_jugadores.layout 
}

# Layout principal de la aplicación
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Callback para proteger las rutas y manejar la navegación
@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    # Si es la página de logout, desconectar al usuario y redirigir a login
    if pathname == '/logout':
        if current_user.is_authenticated:
            logout_user()
        return login.layout
    
    # Si es la raíz, redirigir a dashboard si está autenticado o a login si no
    if pathname == '/':
        if current_user.is_authenticated:
            return dashboard.layout
        else:
            return login.layout
    
    # Si es login y el usuario ya está autenticado, redirigir a dashboard
    if pathname == '/login' and current_user.is_authenticated:
        return dashboard.layout
    
    # Si la ruta existe en nuestro diccionario
    if pathname in pages:
        # Proteger rutas que no son login
        if pathname != '/login' and not current_user.is_authenticated:
            return login.layout
        return pages[pathname]
    
    # Si no existe, mostrar página 404
    return html.Div([
        html.H1("404: Página no encontrada", style={'color': '#00BFFF'}),
        html.P("La página que buscas no existe.", style={'color': 'white'}),
        dcc.Link("Volver al inicio", href="/", className="login-button", style={'display': 'block', 'width': '200px', 'margin': '20px auto'})
    ], className="login-container")

# Ejecutar la aplicación
if __name__ == '__main__':
    # Para desarrollo local:
    if 'RENDER' not in os.environ:
        app.run_server(debug=True)
    # Para Render:
    else:
        # Usa el puerto que Render proporciona y escucha en '0.0.0.0'
        port = int(os.environ.get('PORT', 8080))
        app.run_server(host='0.0.0.0', port=port)