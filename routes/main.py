from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    session,
)

main_bp = Blueprint("main", __name__)


def _get_session_id() -> str:
    if "chat_session_id" not in session:
        session["chat_session_id"] = current_app.chat_memory.new_session_id()
    return session["chat_session_id"]


@main_bp.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@main_bp.route("/api/history", methods=["GET"])
def get_history():
    session_id = _get_session_id()
    messages = current_app.chat_memory.get_history(session_id)
    return jsonify({"messages": messages})


@main_bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required."}), 400

    session_id = _get_session_id()
    chat_service = current_app.chat_service

    try:
        result = chat_service.process_message(session_id, query)
    except Exception as exc:
        current_app.logger.exception("Chat processing failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "answer": result["answer"],
            "chunks": result["chunks"],
            "message": result["message"],
        }
    )


@main_bp.route("/api/clear", methods=["POST"])
def clear_chat():
    session_id = _get_session_id()
    current_app.chat_memory.clear_session(session_id)
    session.pop("chat_session_id", None)
    new_id = _get_session_id()
    return jsonify({"session_id": new_id, "messages": []})
