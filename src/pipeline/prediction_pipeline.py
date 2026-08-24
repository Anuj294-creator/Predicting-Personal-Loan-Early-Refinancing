import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "proccessor.pkl")

           
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            
            scaled_data = preprocessor.transform(features)
            preds = model.predict(scaled_data)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Experience: int,
        Income: float,
        Family: int,
        CCAvg: float,
        Education: int,
        Mortgage: float,
        Securities_Account: int,
        CD_Account: int,
        Online: int,
        CreditCard: int
    ):
        self.Experience = Experience
        self.Income = Income
        self.Family = Family
        self.CCAvg = CCAvg
        self.Education = Education
        self.Mortgage = Mortgage
        self.Securities_Account = Securities_Account
        self.CD_Account = CD_Account
        self.Online = Online
        self.CreditCard = CreditCard

    def get_data_as_data_frame(self):
        try:
            # Map parameters to exact feature names used during model training
            custom_data_input_dict = {
                "Experience": [self.Experience],
                "Income": [self.Income],
                "Family": [self.Family],
                "CCAvg": [self.CCAvg],
                "Education": [self.Education],
                "Mortgage": [self.Mortgage],
                "Securities Account": [self.Securities_Account],
                "CD Account": [self.CD_Account],
                "Online": [self.Online],
                "CreditCard": [self.CreditCard]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)