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



### Project Structure

<table>
  <thead>
    <tr>
      <th>Feature Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Work Experience</b></td>
      <td>Integer</td>
      <td>Years of professional experience (>= 0)</td>
    </tr>
    <tr>
      <td><b>Annual Income</b></td>
      <td>Float</td>
      <td>Annual income in thousands ($000s)</td>
    </tr>
    <tr>
      <td><b>Family Size</b></td>
      <td>Integer</td>
      <td>Number of family members (1 to 10)</td>
    </tr>
    <tr>
      <td><b>Monthly CC Spend</b></td>
      <td>Float</td>
      <td>Average monthly credit card spending ($000s)</td>
    </tr>
    <tr>
      <td><b>Education Level</b></td>
      <td>Categorical</td>
      <td>1: Undergrad, 2: Graduate, 3: Advanced/Professional</td>
    </tr>
    <tr>
      <td><b>Mortgage Value</b></td>
      <td>Float</td>
      <td>Value of house mortgage ($000s)</td>
    </tr>
    <tr>
      <td><b>Securities Account</b></td>
      <td>Binary</td>
      <td>Does customer have a securities account? (0: No, 1: Yes)</td>
    </tr>
    <tr>
      <td><b>CD Account</b></td>
      <td>Binary</td>
      <td>Does customer have a Certificate of Deposit? (0: No, 1: Yes)</td>
    </tr>
    <tr>
      <td><b>Online Banking</b></td>
      <td>Binary</td>
      <td>Does customer use internet banking facilities? (0: No, 1: Yes)</td>
    </tr>
    <tr>
      <td><b>Credit Card</b></td>
      <td>Binary</td>
      <td>Does customer use a credit card issued by bank? (0: No, 1: Yes)</td>
    </tr>
  </tbody>
</table>
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Work Experience** | Integer | Years of professional experience (>= 0) |
| **Annual Income** | Float | Annual income in thousands ($000s) |
| **Family Size** | Integer | Number of family members (1 to 10) |
| **Monthly CC Spend** | Float | Average monthly credit card spending ($000s) |
| **Education Level** | Categorical | 1: Undergrad, 2: Graduate, 3: Advanced/Professional |
| **Mortgage Value** | Float | Value of house mortgage ($000s) |
| **Securities Account** | Binary | Does customer have a securities account? (0: No, 1: Yes) |
| **CD Account** | Binary | Does customer have a Certificate of Deposit? (0: No, 1: Yes) |
| **Online Banking** | Binary | Does customer use internet banking facilities? (0: No, 1: Yes) |
| **Credit Card** | Binary | Does customer use a credit card issued by bank? (0: No, 1: Yes) |




