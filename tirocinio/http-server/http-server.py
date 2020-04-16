from flask import Flask
import socket
app = Flask(__name__)

#app route definisce cosa visualizzare se si accede alla directory del sito
@app.route("/")
def hello():
    return f"Hello World from {socket.gethostname()} \n"

@app.route("/prova")
def easter_egg():
    return "benvenuto nella parte oscura dell'intrnet!"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)