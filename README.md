# Personal Loan Early Refinancing Predictor 🚀

A full-stack Machine Learning application built with Flask and Python that predicts whether a bank customer is likely to accept or qualify for early personal loan refinancing based on their demographic and financial profile.

## 📌 Project Architecture & Directory Structure

```text
├── artifacts/
│   ├── model.pkl               # Trained Machine Learning Model
│   └── processor.pkl           # Data Transformation Pipeline / Scaler
├── templates/
│   └── home.html               # Frontend User Form Interface
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py # CustomData and PredictPipeline logic
│   │   └── train_pipeline.py
│   ├── exception.py            # Custom Exception Handler
│   ├── logger.py               # Custom Logging Utility
│   └── utils.py                # Helper utilities (load_object, save_object)
├── app.py                      # Flask Application Server
├── requirements.txt            # Python Dependencies
└── README.md