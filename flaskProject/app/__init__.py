from flask import Flask
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # # Blueprint 등록
    from .routes import routes_bp
    app.register_blueprint(routes_bp)

    return app
