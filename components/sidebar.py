from dash import html, dcc
import dash_bootstrap_components as dbc

def create_sidebar():
    """
    Crea el componente de la barra lateral de navegación.
    """
    sidebar = html.Div(
        className="sidebar",
        children=[
            html.Div(
                className="app-logo",
                children=[
                    html.H1("SOCCER DATA DBC")
                ]
            ),
            html.Ul(
                className="nav-links",
                children=[
                    html.Li(
                        className="nav-item",
                        children=[
                            dcc.Link(
                                [html.I(className="fas fa-home"), " Dashboard"],
                                href="/dashboard",
                                className="nav-link"
                            )
                        ]
                    ),
                    html.Li(
                        className="nav-item",
                        children=[
                            dcc.Link(
                                [html.I(className="fas fa-users"), " Jugadores Similares"],
                                href="/jugadores-similares",
                                className="nav-link"
                            )
                        ]
                    ),
                    html.Li(
                        className="nav-item",
                        children=[
                            dcc.Link(
                                [html.I(className="fas fa-shield-alt"), " Rendimiento Defensivo"],
                                href="/rendimiento-defensivo",
                                className="nav-link"
                            )
                        ]
                    ),
                    html.Li(
                        className="nav-item",
                        children=[
                            dcc.Link(
                                [html.I(className="fas fa-futbol"), " Rendimiento Ofensivo"],
                                href="/rendimiento-ofensivo",
                                className="nav-link"
                            )
                        ]
                    ),
                    html.Li(
                        className="nav-item",
                        children=[
                            dcc.Link(
                                [html.I(className="fas fa-magic"), " Creación de Juego"],
                                href="/creacion-juego",
                                className="nav-link"
                            )
                        ]
                    ),
                    html.Li(
                        className="nav-item",
                        children=[
                            dcc.Link(
                                [html.I(className="fas fa-running"), " Análisis de Superación"],
                                href="/analisis-superacion",
                                className="nav-link"
                            )
                        ]
                    ),
                    html.Li(
                        className="nav-item",
                        children=[
                            dcc.Link(
                                [html.I(className="fas fa-sign-out-alt"), " Cerrar Sesión"],
                                href="/logout",
                                className="nav-link"
                            )
                        ]
                    ),
                ]
            ),
        ]
    )
    
    return sidebar