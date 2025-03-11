// mobile-performance.js - Mejoras de rendimiento para dispositivos móviles
document.addEventListener('DOMContentLoaded', function() {
    // Detectar si estamos en un dispositivo móvil
    const isMobile = window.innerWidth <= 768;
    
    if (isMobile) {
        // 1. Optimizar la respuesta de los menús y dropdowns
        setTimeout(function() {
            const allDropdowns = document.querySelectorAll('.Select-control');
            allDropdowns.forEach(dropdown => {
                // Mejorar rendimiento de eventos
                dropdown.addEventListener('click', function(e) {
                    e.stopPropagation();
                });
            });
        }, 1000);
        
        // 2. Agregar indicadores de carga para operaciones lentas
        const addLoadingIndicators = () => {
            // Para botones
            const allButtons = document.querySelectorAll('button');
            allButtons.forEach(button => {
                button.addEventListener('click', function() {
                    this.classList.add('loading');
                    this.innerHTML = "Cargando...";
                    
                    // Restaurar texto después de 5 segundos máximo
                    setTimeout(() => {
                        this.classList.remove('loading');
                        this.innerHTML = this.getAttribute('data-original-text') || "BUSCAR";
                    }, 5000);
                    
                    // Guardar texto original
                    if (!this.getAttribute('data-original-text')) {
                        this.setAttribute('data-original-text', this.innerHTML);
                    }
                });
            });
            
            // Para dropdowns
            const allDropdowns = document.querySelectorAll('.Select-control');
            allDropdowns.forEach(dropdown => {
                dropdown.addEventListener('click', function() {
                    const parent = this.closest('.Select');
                    if (parent) {
                        parent.classList.add('is-loading');
                        setTimeout(() => {
                            parent.classList.remove('is-loading');
                        }, 2000);
                    }
                });
            });
        };
        
        // Llamar después de que la interfaz esté lista
        setTimeout(addLoadingIndicators, 2000);
        
        // 3. Optimizar la navegación entre páginas
        const navLinks = document.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                // Mostrar indicador de carga cuando se navega
                document.body.classList.add('page-loading');
                
                // Eliminar después de la carga
                setTimeout(() => {
                    document.body.classList.remove('page-loading');
                }, 3000);
            });
        });
    }
});