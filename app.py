from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
# import sqlite3

app = Flask(__name__)

# Tải mô hình đã lưu
model = joblib.load("credit_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    age = int(request.form['age'])
    income = float(request.form['income'])
    loan_amount = float(request.form['loan_amount'])
    gender = request.form['gender']
    occupation = request.form['occupation']
    
    # Tạo DataFrame từ dữ liệu đầu vào
    new_customer = pd.DataFrame({
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'gender': [gender],
        'occupation': [occupation]
    })
    
    # Bước tiền xử lý dữ liệu để tạo ra các cột mới
    new_customer['income_per_age'] = new_customer['income'] / new_customer['age']
    new_customer['loan_to_income_ratio'] = new_customer['loan_amount'] / new_customer['income']
    
    # Dự đoán A-Score
    a_score = model.predict_proba(new_customer)[:, 1]
    
    # Trả về kết quả dự đoán
    return render_template('index.html', prediction_text=f'A-Score: {a_score[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)