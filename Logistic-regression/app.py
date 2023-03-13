from wsgiref import simple_server
from flask import Flask, request, app
from flask import Response
from flask_cors import CORS
from logistic_deploy import predObj

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True


class ClientApi:

    def __init__(self):
        self.predObj = predObj()

@app.route("/predict", methods=['POST'])
def predictRoute():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is: ', data)
            pred=predObj()
            res = pred.predict_log(data)
            print('result is',res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)


if __name__ == "__main__":
    app.run(debug=True)
    