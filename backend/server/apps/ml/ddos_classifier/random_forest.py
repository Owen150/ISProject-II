# file backend/server/apps/ml/ddos_classifier/random_forest.py
import joblib
import pandas as pd

class RandomForestClassifier:
    #This constructor loads the preprocessing objects as well as the Random Forest object (created with Jupyter notebook)
    def __init__(self):
        path_to_artifacts = "C:/Users/Michael Owen/OneDrive/Desktop/4.2/ISProject-II/notebooks/"
        self.values_fill_missing =  joblib.load(path_to_artifacts + "train_mode.joblib")
        self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "random_forest.joblib")

    #This method takes in the input data in JSON format, converts it to Pandas DataFrame, and applies pre-processing
    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # Filling the missing values
        input_data.fillna(self.values_fill_missing)
        
        return input_data

    #Computing predictions on prepared data
    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    #This method implements post-processing on prediction values
    def postprocessing(self, input_data):
        label = "DDOS"
        if input_data[1] < 0.5:
            label = "BENIGN"
        return {"probability": input_data[1], "label": label, "status": "OK"}

    #This method combines Preprocessing, Predict and Postprocessing methods, and returns a JSON object containing the response
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # Only One Sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
