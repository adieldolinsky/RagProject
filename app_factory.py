import os

from flask import Flask

from config import INSTANCE_FOLDER
from routes.main import main_bp
from services.chat_service import ChatService
from services.memory import ChatMemory


def create_app(vector_store, llm_generator) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or os.urandom(24).hex()

    os.makedirs(INSTANCE_FOLDER, exist_ok=True)

    app.chat_memory = ChatMemory()
    app.chat_service = ChatService(vector_store, llm_generator, app.chat_memory)

    app.register_blueprint(main_bp)

    return app
