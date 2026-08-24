from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('home.html')  # Changed from index.html to home.html

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Changed from index.html to home.html
    else:
        form_data = request.form
        data = CustomData(
            Experience=int(request.form.get('Experience')),
            Income=float(request.form.get('Income')),
            Family=int(request.form.get('Family')),
            CCAvg=float(request.form.get('CCAvg')),
            Education=int(request.form.get('Education')),
            Mortgage=float(request.form.get('Mortgage')),
            Securities_Account=int(request.form.get('Securities_Account')),
            CD_Account=int(request.form.get('CD_Account')),
            Online=int(request.form.get('Online')),
            CreditCard=int(request.form.get('CreditCard'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        output = "Eligible / Likely to Refinance" if results[0] == 1 else "Not Eligible / Unlikely to Refinance"
        
        return render_template('home.html', results=output)  # Changed from index.html to home.html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)