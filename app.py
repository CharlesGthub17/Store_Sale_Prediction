from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

trained_model=joblib.load("C:\\Users\\charl\\Desktop\\fsproject\\finalized_modelSP1.sav")

@app.route("/", methods=["GET","POST"])
def basic():
    if request.method == "POST":
        Quantity = int(request.form["Quantity"])
        Unit_Price = float(request.form["Unit Price"])
        Date_Day = int(request.form["Date Day"])
        Casual_Shirts = int(request.form["Casual Shirts"])
        Coats = int(request.form["Coats"])
        Cycling_Jerseys = int(request.form["Cycling Jerseys"])
        Dress = int(request.form["Dress"])
        Formal_Shirts = int(request.form["Formal Shirts"])
        GolfShoes = int(request.form["GolfShoes"])
        Jeans = int(request.form["Jeans"])
        Knitwear = int(request.form["Knitwear"])
        Pants = int(request.form["Pants"])
        Polo_Shirts = int(request.form["Polo Shirts"])
        Pyjamas = int(request.form["Pyjamas"])
        Shorts = int(request.form["Shorts"])
        Suits = int(request.form["Suits"])
        Sweats = int(request.form["Sweats"])
        Ties = int(request.form["Ties"])
        Tshirts = int(request.form["Tshirts"])
        Underwear = int(request.form["Underwear"])

        y_pred = np.array([[Quantity, Unit_Price, Date_Day,Casual_Shirts, Coats, Cycling_Jerseys, Dress,Formal_Shirts, GolfShoes, Jeans, Knitwear,Pants, Polo_Shirts, Pyjamas, Shorts,Suits, Sweats, Ties, Tshirts, Underwear]])
        prediction_value =trained_model.predict(y_pred)
        return render_template("index.html", prediction_value = prediction_value[0])
    else:
        return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)