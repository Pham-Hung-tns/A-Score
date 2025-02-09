# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Tạo dữ liệu mẫu ban đầu
data = pd.DataFrame({
    'age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 55],
    'income': [50000, 100000, 75000, 120000, 45000, 80000, 110000, 95000, 70000, 105000],
    'loan_amount': [20000, 50000, 30000, 60000, 15000, 35000, 55000, 40000, 25000, 45000],
    'gender': ['male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female'],
    'occupation': ['engineer', 'doctor', 'lawyer', 'engineer', 'teacher', 'lawyer', 'doctor', 'teacher', 'engineer', 'lawyer'],
    'target': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
})

# Tạo thêm dữ liệu mẫu
np.random.seed(42)
n_samples = 20000

ages = np.random.randint(18, 70, n_samples)
incomes = np.random.randint(20000, 150000, n_samples)
loan_amounts = np.random.randint(5000, 100000, n_samples)
genders = np.random.choice(['male', 'female'], n_samples)
occupations = np.random.choice(['engineer', 'doctor', 'lawyer', 'teacher'], n_samples)
targets = np.random.choice([0, 1], n_samples)

new_data = pd.DataFrame({
    'age': ages,
    'income': incomes,
    'loan_amount': loan_amounts,
    'gender': genders,
    'occupation': occupations,
    'target': targets
})

# Kết hợp dữ liệu mẫu ban đầu và dữ liệu mới
combined_data = pd.concat([data] + [new_data], ignore_index=True)

# Bước 1: Xử lý các giá trị thiếu (nếu có)
imputer = SimpleImputer(strategy='median')
combined_data[['age', 'income', 'loan_amount']] = imputer.fit_transform(combined_data[['age', 'income', 'loan_amount']])

# Bước 2: Tạo các biến mới
# Ví dụ: Tạo biến 'income_per_age' và 'loan_to_income_ratio'
combined_data['income_per_age'] = combined_data['income'] / combined_data['age']
combined_data['loan_to_income_ratio'] = combined_data['loan_amount'] / combined_data['income']

# Bước 3: Mã hóa các biến phân loại
categorical_features = ['gender', 'occupation']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # B3.1: Xử lý giá trị thiếu
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # B3.2: Mã hóa one-hot: chuyển đổi dữ liệu dạng categorical (danh mục) thành dạng số
])
# -> Nếu số lượng giá trị danh mục quá lớn, có thể dùng Target Encoding hoặc Embedding thay vì One-Hot Encoding.

# Chuẩn hóa các biến số
# -- StandardScaler: chuẩn hóa dữ liệu theo phân phối chuẩn (mean = 0, std = 1)
numerical_features = ['age', 'income', 'loan_amount', 'income_per_age', 'loan_to_income_ratio']
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Kết hợp các bước tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Bước 4: Chuẩn bị dữ liệu
X = combined_data.drop('target', axis=1)  # Các đặc trưng
y = combined_data['target']  # Nhãn mục tiêu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# -> test_size = 0.2: 20% dữ liệu sẽ được dùng làm tập kiểm tra, còn 80% dữ liệu sẽ dùng để huấn luyện mô hình.
# -> random_state: để cố định việc chia dữ liệu, giúp kết quả của mô hình ổn định qua các lần chạy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 5: Tạo mô hình XGBoost với pipeline
# -> logloss là hàm mất mát cho bài toán phân loại nhị phân (Binary Classification). eval_metric: chọn hàm đánh giá mô hình.
# -> use_label_encoder=False: để tránh cảnh báo về việc sử dụng label encoder.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Bước 6: Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Bước 7: Tính toán A-Score
# Giả sử bạn muốn tính A-Score cho một khách hàng mới
new_customer = pd.DataFrame({
    'age': [30],'income': [50000],
    'loan_amount': [20000],
    'gender': ['male'],
    'occupation': ['engineer'],
    'income_per_age': [50000 / 30],
    'loan_to_income_ratio': [20000 / 50000]
})

# -> hàm predict_proba sẽ chạy đúng số lần bằng số lượng mẫu đầu vào. trong trường hợp này, chỉ có 1 mẫu nên cần lấy phần tử đầu tiên.
# -> Số hàng = số lượng khách hàng cần dự đoán. Số cột = số lớp trong nhãn mục tiêu (y_train)
a_score = model.predict_proba(new_customer)[:, 1]
print(f'A-Score: {a_score[0]:.2f}')


import joblib

# Lưu mô hình đã huấn luyện
joblib.dump(model, "credit_model.pkl")
print("Model saved to credit_model.pkl")


