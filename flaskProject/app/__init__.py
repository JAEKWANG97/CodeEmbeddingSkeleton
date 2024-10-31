# flaskProject/app/__init__.py
from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()  # ...env 파일에서 환경 변수 로드


def create_app():
    app = Flask(__name__)
    app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    from .routes import routes_bp
    app.register_blueprint(routes_bp)

    return app
