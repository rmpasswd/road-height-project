import numpy as np
from flask import Flask as fl
from flask import request, render_template
from sklearn.preprocessing import scale                                                          

import pickle


app= fl(__name__)
model= pickle.load(open('./model.pkl','rb'))



@app.route('/')
def homepage():
	return render_template('./index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

	int_features = [int(request.form['day']), \
		float(request.form['waterlevel']), float(request.form['rainfall'])]
	print(int_features)
	final_features = np.array(int_features+\
		[int_features[0]*int_features[1],int_features[0]*int_features[2],int_features[1]*int_features[2]])

	final_features_scaled= scale(final_features)
	prediction = model.predict([final_features_scaled])
	print(final_features_scaled)
	print("prediction value: ",prediction)

	if prediction[0]==1:
		return render_template('index.html', prediction_text=f"There is a high chance of flooding on Day {int_features[0]}, WaterLevel {int_features[1]} , Rainfall {int_features[2]}")

	else:
		return render_template('index.html', prediction_text=f"No chance of flooding on Day {int_features[0]}, WaterLevel {int_features[1]} , Rainfall {int_features[2]}")
