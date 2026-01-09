from flask import Flask, render_template, request
from engine.ml_engine import run_all_ml

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        password = request.form["password"]
        result = run_all_ml(password)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
