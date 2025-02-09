import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

# Tạo dữ liệu mẫu ban đầu
data = pd.DataFrame({
    'age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 55],
    'income': [50000, 100000, 75000, 120000, 45000, 80000, 110000, 95000, 70000, 105000],
    'loan_amount': [20000, 50000, 30000, 60000, 15000, 35000, 55000, 40000, 25000, 45000],
    'gender': ['male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female'],
    'occupation': ['engineer', 'doctor', 'lawyer', 'engineer', 'teacher', 'lawyer', 'doctor', 'teacher', 'engineer', 'lawyer'],
    'target': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
})

# Tạo thêm dữ liệu mẫu với các feature mới
np.random.seed(42)
n_samples = 20000

ages = np.random.randint(18, 70, n_samples)
incomes = np.random.randint(20000, 150000, n_samples)
loan_amounts = np.random.randint(5000, 100000, n_samples)
genders = np.random.choice(['male', 'female'], n_samples)
occupations = np.random.choice(['engineer', 'doctor', 'lawyer', 'teacher'], n_samples)
targets = np.random.choice([0, 1], n_samples)

# Các feature mới
home_ownerships = np.random.choice(['rent', 'mortgage', 'own'], n_samples)
emp_lengths = np.random.randint(0, 40, n_samples)
loan_intents = np.random.choice(['education', 'medical', 'venture'], n_samples)
loan_int_rates = np.random.uniform(5.0, 20.0, n_samples)
loan_percent_incomes = loan_amounts / incomes
default_records = np.random.choice(['Y', 'N'], n_samples)
cred_hist_lengths = np.random.randint(1, 30, n_samples)
terms = np.random.choice(['36 months', '60 months'], n_samples)
int_rates = np.random.uniform(5.0, 20.0, n_samples)
installments = loan_amounts / np.array([36 if term == '36 months' else 60 for term in terms])
grades = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
member_ids = np.arange(1, n_samples + 1)

new_data = pd.DataFrame({
    'age': ages,
    'income': incomes,
    'loan_amount': loan_amounts,
    'gender': genders,
    'occupation': occupations,
    'target': targets,
    # Các feature mới
    'person_home_ownership': home_ownerships,
    'person_emp_length': emp_lengths,
    'loan_intent': loan_intents,
    'loan_int_rate': loan_int_rates,
    'loan_percent_income': loan_percent_incomes,
    'cb_person_default_on_file': default_records,
    'cb_person_cred_hist_length': cred_hist_lengths,
    'term': terms,
    'int_rate': int_rates,
    'installment': installments,
    'grade': grades,
    'member_id': member_ids
})

# Kết hợp dữ liệu mẫu ban đầu và dữ liệu mới
combined_data = pd.concat([data, new_data], ignore_index=True)
# Bước tiền xử lý dữ liệu

# Xử lý các giá trị thiếu (nếu có)
imputer = SimpleImputer(strategy='median')
combined_data[['age', 'income', 'loan_amount']] = imputer.fit_transform(combined_data[['age', 'income', 'loan_amount']])

# Tạo các biến mới
combined_data['income_per_age'] = combined_data['income'] / combined_data['age']
combined_data['loan_to_income_ratio'] = combined_data['loan_amount'] / combined_data['income']

# Xử lý các giá trị NaN trong cột 'term' và 'installment'
combined_data['term'] = combined_data['term'].fillna('36 months')
combined_data['installment'] = combined_data['installment'].fillna(combined_data['loan_amount'] / 36)

# Các biến mới nâng cao
combined_data['annual_installment'] = combined_data['installment'] * combined_data['term'].str.extract(r'(\d+)').astype(float)[0] / 12
combined_data['debt_to_income_ratio'] = combined_data['loan_amount'] / combined_data['income']
combined_data['credit_history_per_age'] = combined_data['cb_person_cred_hist_length'] / combined_data['age']

# Mã hóa các biến phân loại
categorical_features = ['gender', 
                        'occupation',
                        'person_home_ownership',
                        'loan_intent',
                        'cb_person_default_on_file',
                        'term',
                        'grade']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Chuẩn hóa các biến số
numerical_features = ['age',
                      'income',
                      'loan_amount',
                      # Các biến mới tạo ra
                      'income_per_age',
                      'loan_to_income_ratio',
                      # Các biến khác
                      'person_emp_length',
                      'loan_int_rate',
                      # Các biến nâng cao
                      'annual_installment',
                      'debt_to_income_ratio',
                      # Các biến khác
                      'loan_percent_income',
                      # Các biến nâng cao
                      'credit_history_per_age',
                      # Các biến khác
                      'cb_person_cred_hist_length',
                      # Các biến khác
                      'int_rate',# Các biến khác
                      'installment']
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Kết hợp các bước tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Chuẩn bị dữ liệu
X = combined_data.drop(['target'], axis=1) # Các đặc trưng
y = combined_data['target'] # Nhãn mục tiêu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình XGBoost với pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Lưu mô hình đã huấn luyện
joblib.dump(model, "credit_model.pkl")
print("Model saved to credit_model.pkl")



# =============================================================================
# Giải thích về từng trường trong dữ liệu JSON mà bạn đã sử dụng:
#   1. age: Tuổi của người nộp đơn.
#   2. income: Thu nhập hàng năm của người nộp đơn.
#   3. loan_amount: Số tiền vay.
#   4. gender: Giới tính của người nộp đơn (ví dụ: "male" hoặc "female").
#   5. occupation: Nghề nghiệp của người nộp đơn (ví dụ: "engineer", "doctor", "lawyer", "teacher").
#   6. person_home_ownership: Loại hình sở hữu nhà của người nộp đơn (ví dụ: "rent", "mortgage", "own").
#   7. person_emp_length: Thời gian làm việc (tính bằng năm) của người nộp đơn.
#   8. loan_intent: Mục đích của khoản vay (ví dụ: "education", "medical", "venture").
#   9. loan_int_rate: Lãi suất của khoản vay.
#   10. loan_percent_income: Tỷ lệ phần trăm thu nhập dành cho khoản vay.
#   11. cb_person_default_on_file: Hồ sơ vỡ nợ lịch sử (ví dụ: "Y" cho có, "N" cho không).
#   12. cb_person_cred_hist_length: Thời gian lịch sử tín dụng (tính bằng năm).
#   13. term: Thời hạn của khoản vay (ví dụ: "36 months", "60 months").
#   14. int_rate: Lãi suất của khoản vay.
#   15. installment: Khoản trả góp hàng tháng.
#   16. grade: Xếp hạng tín dụng của người nộp đơn (ví dụ: "A", "B", "C", "D").
#   17. member_id: ID thành viên của người nộp đơn.

# Biến mục tiêu trong dữ liệu của bạn là target. Đây là biến mà mô hình của bạn sẽ dự đoán. Ý nghĩa của biến mục tiêu này là xác định xem một khoản vay có bị vỡ nợ hay không. Cụ thể:
#   • 0: Khoản vay không bị vỡ nợ (người vay đã trả nợ đúng hạn).
#   • 1: Khoản vay bị vỡ nợ (người vay không trả nợ đúng hạn).
# Mục tiêu của mô hình là dự đoán xác suất vỡ nợ của một khoản vay mới dựa trên các đặc trưng đầu vào như tuổi, thu nhập, số tiền vay, nghề nghiệp, và các yếu tố khác. Điều này giúp các tổ chức tài chính đánh giá rủi ro và quyết định có nên cấp khoản vay cho một khách hàng cụ thể hay không.
# =============================================================================









