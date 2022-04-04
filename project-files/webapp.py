import numpy as np
from flask import Flask 
from flask import request, render_template
from sklearn.preprocessing import scale                                                          
from datetime import date
import pickle,sys

import waterlevelffwc,getrainfallffwc

app= Flask(__name__)
model= pickle.load(open('./model.pkl','rb'))



@app.route('/')
def homepage():
	return render_template('./index.html')	#when user go to homescreen

@app.route('/predict',methods=['POST','GET'])	# the form inputs can be accessed by dictionary named request.forms['']
def predict():

	if bool(request.form['waterlevel']) is False:
		waterlevel =  waterlevelffwc.wtrlvl2
	else:
		waterlevel = request.form['waterlevel']

	if bool(request.form['date']) is False:
		datevalue=  str(date.today())
	else:
		datevalue = request.form['date']


	if bool(request.form['rainfall']) is False:
		rainfall =  getrainfallffwc.rainfall
	else:
		rainfall=request.form['rainfall']

	# print("length:",len(request.form), "date", request.form['date'])
	datesplit=datevalue.split("-")	
	
	datercvd = date(int(datesplit[0]), int(datesplit[1]), int(datesplit[2])) # we have to take the date(2022-03-31) and make a day
	daycount = int(datercvd.strftime('%j'))
	
	int_features = [ daycount, \
		float(waterlevel), float(rainfall)] # int_features array will contain the 3 inputs
	print(int_features)


	final_features = np.array(int_features+\
		[int_features[0]*int_features[1],int_features[0]*int_features[2],int_features[1]*int_features[2]])	# converted to a numpy array
	
	final_features_scaled= scale(final_features)
	prediction = model.predict([final_features_scaled]) # passed to the model file.
	
	print(final_features_scaled)
	print("prediction value: ",prediction)

	if prediction[0]==1:
		return render_template('index.html', prediction_text=f"There is a high chance of flooding on  {datevalue}, WaterLevel {int_features[1]} , Rainfall {int_features[2]}")

	else:
		return render_template('index.html', prediction_text=f"No chance of flooding on Day {int_features[0]}, WaterLevel {int_features[1]} , Rainfall {int_features[2]}")

if __name__ == '_main_':
	app.debug = True
	app.run(host = "1.2.3.4", port = 5000)