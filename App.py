from flask import Flask, Response
from flask_cors import CORS
import requests as requests
import json
import pytchat
import pafy
import re

app = Flask(__name__)
CORS(app)


running = True


@app.route('/<BCID>/<Email>')
def sse(BCID, Email):
    def generate(BCID, Email):
        chat = pytchat.create(video_id=BCID)
        print(Email)
        youtube_api_key = ""
        client_id = ""
        client_secret = ""
        pafy.set_api_key(youtube_api_key)
        url = "https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/json"
        }
        preName = ""
        preDate = ""
        while True:
            try:
                data = chat.get()
                items = data.items
                for c in items:
                    if not (preDate == c.datetime and preName == c.author.name):
                        print(c.author.name, " : ", c.message)
                        message = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", c.message)
                        print(c.author.name," : ",message)

                        data = {"content": message}
                        response = requests.post(url, data=json.dumps(data), headers=headers)

                        text = response.json()
                        header = {"Content-type": "application/json", "Accept": "text/plain"}
                        if 'sentences' in text:
                            sen = text['sentences'][0]
                            data2 = {
                                "author": c.author.name,
                                "dateTime": c.datetime,
                                "message": message,
                                "emotion3": jsonmax(sen['confidence']),
                                "emotion7": 2
                            }
                            yield f"data:{data2}\n\n"
                            requests.get(
                                "http://localhost:8080/chat?email=" + Email + "&BCID=" + BCID + "&name=" + c.author.name,
                                data=json.dumps(data2), headers=header)
                        else:
                            data2 = {
                                "author": c.author.name,
                                "dateTime": c.datetime,
                                "message": message,
                                "emotion3": 2,
                                "emotion7": 2
                            }

                            f"data:{data2}\n\n"
                            URI = "http://localhost:8080/chat?email=" + Email + "&BCID=" + BCID + "&name=" + c.author.name
                            response = requests.get(URI, data=json.dumps(data2), headers=header)

                        preName = c.author.name
                        preDate = c.datetime
            except KeyboardInterrupt:
                break

    return Response(generate( BCID, Email), mimetype='text/event-stream')

def jsonmax(data):
    max_value = max(data.values())
    index = 0
    for key, value in data.items():
        if value == max_value:
            return index
        index += 1


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=9900, threaded=False, processes=10)
