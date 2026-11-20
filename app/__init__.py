"""CVInsight Application Factory."""
import os
from flask import Flask
from app.config import settings
from app.database.repository import init_db


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
    )
    app.config["UPLOAD_FOLDER"] = settings.upload_folder
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    init_db()

    from app.routes.matcher import matcher_bp
    from app.routes.chat import chat_bp

    app.register_blueprint(matcher_bp)
    app.register_blueprint(chat_bp)

    return app
