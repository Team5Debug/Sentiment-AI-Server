from Training import *


#질문 무한반복하기! 0 입력시 종료

from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


running = True
print("실행레쓰고")

@app.route('/ai/<sen>')
def aa(sen):

    result = predict(sen)
    return f"{result}"



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=2942)
