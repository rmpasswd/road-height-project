from flask import Flask
import pickle

model= pickle.load(open('./model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '_main_':
    app.run()