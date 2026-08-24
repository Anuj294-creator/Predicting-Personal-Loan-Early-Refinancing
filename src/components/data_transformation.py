import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'proccessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        
        try:
            # All predictor features after dropping ID, Age, ZIP Code, and Personal Loan
            numerical_columns = [
                'Experience',
                'Income',
                'Family',
                'CCAvg',
                'Education',
                'Mortgage',
                'Securities Account',
                'CD Account',
                'Online',
                'CreditCard'
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", StandardScaler(), numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Personal Loan"
            drop_columns = ["ID", "ZIP Code", "Age", target_column_name]

           
            train_df['Experience'] = train_df['Experience'].apply(lambda x: 0 if x < 0 else x)
            test_df['Experience'] = test_df['Experience'].apply(lambda x: 0 if x < 0 else x)

           
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

         
            input_feature_train_scaled = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_scaled = preprocessing_obj.transform(input_feature_test_df)

           
            logging.info("Applying SMOTE on scaled training data")
            smote = SMOTE(random_state=42)
            input_feature_train_resampled, target_feature_train_resampled = smote.fit_resample(
                input_feature_train_scaled, target_feature_train_df
            )

            
            train_arr = np.c_[
                input_feature_train_resampled, np.array(target_feature_train_resampled)
            ]
            test_arr = np.c_[
                input_feature_test_scaled, np.array(target_feature_test_df)
            ]

            
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)