import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42)
            }

            # Optional hyperparameter grid dictionary if evaluate_models expects params
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10, 20]
                },
                "Logistic Regression": {}
            }

            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models,
                param=params
)
           
            best_model_score = max(sorted(model_report.values()))

            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)

            logging.info(f"Best found model on testing dataset: {best_model_name} with score: {best_model_score}")

            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            
            
            acc_val = accuracy_score(y_test, predicted)
            f1_val = f1_score(y_test, predicted)
            precision_val = precision_score(y_test, predicted)
            recall_val = recall_score(y_test, predicted)
            roc_auc_val = roc_auc_score(y_test, predicted)

            return {
                "model_name": best_model_name,
                "model_object": best_model,
                "accuracy_score": acc_val,
                "f1_score": f1_val,
                "precision_score": precision_val,
                "recall_score": recall_val,
                "roc_auc_score": roc_auc_val
            }

        except Exception as e:
            raise CustomException(e, sys)