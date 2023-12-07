from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

with open('Logistic_Regression_Refined.p', 'rb') as f2:
    print("utilisation modele Logistic Regression")
    grid_lgbm = pickle.load(f2)

with open('explainer.p', 'rb') as f1:
    print("Shap modele explainer")
    explainer = pickle.load(f1)

df = pd.read_csv('test_data.csv', index_col=0)
df=df.reset_index()
num_client = df.SK_ID_CURR.unique()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/')
def predict():
    """

    Returns
    liste des clients dans le fichier

    """
    return jsonify({"model": "Logistic_Regression",
                    "list_client_id" :  list(num_client.astype(str))})




@app.route('/predict/<int:sk_id>')
def predict_get(sk_id):
    """

    Parameters
    ----------
    sk_id : numero de client

    Returns
    -------
    prediction  0 pour paiement OK
                1 pour defaut de paiement

    """
    if sk_id in num_client:
        client_info=df[df.SK_ID_CURR==sk_id]
        col=list(client_info.columns)
        col.remove("SK_ID_CURR")
        client_info=client_info[col]
        shap_values = explainer.shap_values(client_info)
        print(shap_values)
        predict = grid_lgbm.predict(client_info)[0]
        predict_proba = grid_lgbm.predict_proba(client_info)[0]
        predict_proba_0 = str(predict_proba[0])
        predict_proba_1 = str(predict_proba[1])
    else:
        predict = predict_proba_0 = predict_proba_1 = "client inconnu"
    return jsonify({ 'retour_prediction' : str(predict), 'predict_proba_0': predict_proba_0,
                     'predict_proba_1': predict_proba_1,
                      'shap_values': shap_values.tolist()})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
					 