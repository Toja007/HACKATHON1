import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend like Agg
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder




app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    # Render the HTML page
    return render_template('frontend.html')

# Load pre-trained model (ensure your stacked model is saved as a .pkl file)
model_path = 'stacked_model.pkl'
model = joblib.load(model_path)

# Helper function to calculate average scores
def calculate_averages(df):
    score_columns = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
    df['average_score'] = df[score_columns].mean(axis=1)
    return df

# Helper function to generate statistics and insights
def generate_statistics(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['average_score'], kde=True, bins=10)
    plt.title("Distribution of Average Scores")
    chart_path = os.path.join("static", "average_score_chart.png")
    plt.savefig(chart_path)
    plt.close()
    
    stats = df.describe()  # Summary statistics
    return stats, chart_path

# Improvement suggestions based on certain conditions
def suggest_improvements(row):
    suggestions = []
    if row['part_time_job']:
        suggestions.append("Consider reducing part-time work hours to focus more on studies.")
    if row['absence_days'] > 5:
        suggestions.append("Improve attendance.")
    if not row['extracurricular_activities']:
        suggestions.append("Consider participating in extracurricular activities to develop soft skills.")
    if row['weekly_self_study_hours'] <= 7:
        suggestions.append("Increase self-study hours to at least 10 hours per week.")

    return suggestions

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)

        # Calculate average score
        df = calculate_averages(df)
        
        # Predict whether students will pass
        prediction_features = ['part_time_job', 'absence_days', 'extracurricular_activities', 'weekly_self_study_hours',
            'math_score',
       'history_score', 'physics_score', 'chemistry_score', 'biology_score',
       'english_score', 'geography_score']
        le = LabelEncoder()
        df['part_time_job'] = le.fit_transform(df['part_time_job'])
        df['extracurricular_activities'] = le.fit_transform(df['extracurricular_activities'])
        scaler = StandardScaler()
        df[prediction_features] = scaler.fit_transform(df[prediction_features])
        df['pass_prediction'] = model.predict(df[prediction_features])
        
        # Generate statistics
        stats, chart_path = generate_statistics(df)
        
        # For students predicted not to pass, suggest improvements
        df['improvement_suggestions'] = df.apply(lambda row: suggest_improvements(row) if row['pass_prediction'] == 0 else '', axis=1)

        # Respond with predictions, stats, and chart
        return jsonify({
            "predictions": df[['id', 'first_name', 'last_name', 'pass_prediction', 'improvement_suggestions']].to_dict(orient='records'),
            "statistics": stats.to_dict(),
            "chart_url": chart_path
        })

    return jsonify({"error": "File type not supported"}), 400

if __name__ == '__main__':
    app.run(debug=True)
