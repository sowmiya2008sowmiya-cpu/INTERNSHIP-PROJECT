from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and trained column names
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['Gender']
        age = float(request.form['Age'])
        attendance = float(request.form['Attendance'])
        study_hours = float(request.form['Study_Hours'])
        parental_edu = request.form['Parental_Education']
        prev_score = float(request.form['Previous_Score'])
        
        math_score = float(request.form['Math_Score'])
        science_score = float(request.form['Science_Score'])
        english_score = float(request.form['English_Score'])
        
        internet = request.form['Internet_Access']
        extracurricular = request.form['Extracurricular']
        socio_economic = request.form['Socio_Economic']
        school_type = request.form['School_Type']

        # Manual One-Hot Encoding mapping exactly matching your CSV columns
        input_dict = {
            'age': age,
            'attendance_percent': attendance,
            'study_hours_per_week': study_hours,
            'previous_exam_score': prev_score,
            'math_score': math_score,
            'science_score': science_score,
            'english_score': english_score,
            'gender_Male': 1 if gender == 'Male' else 0,
            'parental_education_High school': 1 if parental_edu == 'High school' else 0,
            'parental_education_No formal': 1 if parental_edu == 'No formal' else 0,
            'parental_education_Postgraduate': 1 if parental_edu == 'Postgraduate' else 0,
            'parental_education_Undergraduate': 1 if parental_edu == 'Undergraduate' else 0,
            'internet_access_Yes': 1 if internet == 'Yes' else 0,
            'extracurricular_activities_Yes': 1 if extracurricular == 'Yes' else 0,
            'socio_economic_status_Low': 1 if socio_economic == 'Low' else 0,
            'socio_economic_status_Medium': 1 if socio_economic == 'Medium' else 0,
            'socio_economic_status_High': 1 if socio_economic == 'High' else 0,
            'school_type_Public': 1 if school_type == 'Public' else 0
        }

        input_data = pd.DataFrame([input_dict])
        
        # Reindex to strictly match the columns the model was trained on
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        
        # --- DEBUGGING LINES ---
        print("--- DEBUGGING INPUT DATA ---")
        print(input_data.to_string()) # Shows the exact values being sent to model
        print("----------------------------")
        
        # Predict the performance level
        prediction = model.predict(input_data)[0]
        print("Model Output is:", prediction)

        # Retain original inputs in UI
        return render_template('index.html', 
                               prediction=prediction,
                               Gender=gender, Age=age, Attendance=attendance,
                               Study_Hours=study_hours, Parental_Education=parental_edu,
                               Previous_Score=prev_score, Math_Score=math_score,
                               Science_Score=science_score, English_Score=english_score,
                               Internet_Access=internet, Extracurricular=extracurricular,
                               Socio_Economic=socio_economic, School_Type=school_type)

if __name__ == '__main__':
    app.run(debug=True)