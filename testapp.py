import unittest
from flask import Flask
from app import predict_get,predict
import pickle
import pandas as pd



df = pd.read_csv('test_data.csv', index_col=0)
df=df.reset_index()
num_client = df.SK_ID_CURR.unique()

class TestPredictGet(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['DEBUG'] = False

    def test_predict(self):
        with self.app.test_request_context('/predict/'):
            response = predict()

        # Vérifiez le code de statut de la réponse
        self.assertEqual(response.status_code, 200)

        # Vérifiez le contenu JSON de la réponse
        data = response.get_json()
        self.assertIn('model', data)
        self.assertIn('list_client_id', data)

    def test_predict_get_client_exist(self):
        num_client_existants = list(num_client)[0]
 
        with self.app.test_request_context(f'/predict/{int(num_client_existants)}'):
            response = predict_get(int(num_client_existants))
        

        self.assertEqual(response.status_code, 200) 


        data = response.get_json()
        self.assertIn('retour_prediction', data)
        self.assertIn('predict_proba_0', data)
        self.assertIn('predict_proba_1', data)
        self.assertIn('shap_values', data)

    def test_predict_get_client_inconnu(self):
        num_client_existants = '99'
        
        with self.app.test_request_context(f'/predict/{int(num_client_existants)}'):
            response = predict_get(int(num_client_existants))
        

        self.assertEqual(response.status_code, 200)

        # Vérifiez le contenu JSON de la réponse pour un client inconnu
        data = response.get_json()
        self.assertEqual(data['retour_prediction'], 'client inconnu')

if __name__ == '__main__':
    unittest.main()
