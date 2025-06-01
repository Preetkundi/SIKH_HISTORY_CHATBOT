from flask import Flask, request, render_template, jsonify
from pipeline import answer_punjabi_question

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question", "")
    answer = answer_punjabi_question(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
