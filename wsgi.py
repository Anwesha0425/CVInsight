"""Production WSGI entry point for gunicorn / uWSGI."""
from app import create_app

application = create_app()
app = application  # alias for compatibility
