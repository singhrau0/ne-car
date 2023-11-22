from flask import Flask, render_template,jsonify,request
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method=="POST":
        make = request.form.get("make")
        model = request.form.get("model")
        year = request.form.get("year")
        fuel_type = request.form.get("fuel_type")
        hp = request.form.get("hp")
        cylinders = request.form.get("cylinders")
        transmission = request.form.get("transmission")
        driven_wheels = request.form.get("driven_wheels")
        doors = request.form.get("doors")
        size = request.form.get("size")
        style = request.form.get("style")
        highway_mpg = request.form.get("highway_mpg")
        city_mpg = request.form.get("city_mpg")
        popularity = request.form.get("popularity")
        df = pd.read_json("new.json")
        make_encode = df['Make_encode'][df['Make']==make].values[0]
        model_encode = df['Model_encode'][df['Model']==model].values[0]
        eft_encode = df['Engine Fuel Type_encode'][df['Engine Fuel Type']==fuel_type].values[0]
        dw_encode = df['Driven_Wheels_encode'][df['Driven_Wheels']==driven_wheels].values[0]
        tt_encode = df['Transmission Type_encode'][df['Transmission Type']==transmission].values[0]
        vs_encode = df['Vehicle Style_encode'][df['Vehicle Style']==style].values[0]
        vsz_encode = df['Vehicle Size_encode'][df['Vehicle Size']==size].values[0]
        print(make_encode,model_encode,eft_encode,dw_encode,tt_encode,vs_encode,vsz_encode)
        with open('model.pkl','rb') as model:
            mlmodel = pickle.load(model)
        predict = mlmodel.predict([[int(year),float(hp),float(cylinders),float(doors),float(highway_mpg),float(city_mpg),float(popularity),make_encode,model_encode,eft_encode,tt_encode,dw_encode,vsz_encode,vs_encode]])
        print(predict)
        print(make,model,year,fuel_type,hp,cylinders,transmission,driven_wheels,doors,size,style,highway_mpg,city_mpg,popularity)
        return jsonify({"Predicted Result":f"result :{predict}"})
    else:
        return render_template("predict.html")




if __name__=='__main__':
    app.run(host = '0.0.0.0')
