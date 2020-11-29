from flask import Flask, render_template, request, url_for
import os
import pickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/result', methods = ['POST'])
def result():
    Pregnancies = float(request.form.get('Pregnancies'))
    Glucose = float(request.form.get('Glucose'))
    Blood_Pressure = float(request.form.get('Blood Pressure'))
    Skin_Thickness = float(request.form.get('Skin Thickness'))
    Insulin = float(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    Diabetes_Pedigree_Function = float(request.form.get('Diabetes Pedigree Function'))
    Age = float(request.form.get('Age'))

    log_reg = pickle.load(open('modelForPrediction.sav', 'rb'))
    scaler = StandardScaler()
    data = [[Pregnancies, Glucose, Blood_Pressure, Skin_Thickness, Insulin,BMI,  Diabetes_Pedigree_Function, Age]]
    result = log_reg.predict(data)[0]

    if result==0:
        return render_template('result.html', result = 'Your results are Negative..')
    else:
        return render_template('result.html', result = 'Your results are Positive. Sorry!')


@app.route('/help', methods = ['GET'])
def help():
    return render_template('help.html')


@app.route('/about', methods = ['GET'])
def about():
    return render_template('about.html')



# port = int(os.getenv("PORT"))
if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=port)