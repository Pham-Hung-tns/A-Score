from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Tải mô hình đã lưu
model = joblib.load("credit_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Dự đoán A-Score cho khách hàng mới
    """
    # Lấy dữ liệu từ request
    data = request.get_json(force=True)
    print(data)
    
    # Chuyển đổi dữ liệu thành DataFrame
    new_customers = pd.DataFrame(data)
    
    # Bước tiền xử lý dữ liệu để tạo ra các cột mới
    new_customers['income_per_age'] = new_customers['income'] / new_customers['age']
    new_customers['loan_to_income_ratio'] = new_customers['loan_amount'] / new_customers['income']
    new_customers['annual_installment'] = new_customers['installment'] * new_customers['term'].str.extract(r'(\d+)').astype(int)[0] / 12
    new_customers['debt_to_income_ratio'] = new_customers['loan_amount'] / new_customers['income']
    new_customers['credit_history_per_age'] = new_customers['cb_person_cred_hist_length'] / new_customers['age']
    
    # Dự đoán A-Score
    a_scores = model.predict_proba(new_customers)[:, 1]
    
    # Trả về kết quả dự đoán
    return jsonify([{'A-Score': round(float(score), 2)} for score in a_scores])

if __name__ == '__main__':
    app.run(debug=True)





# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# app = Flask(__name__)

# # Tải mô hình đã lưu
# model = joblib.load("credit_model.pkl")

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Dự đoán A-Score cho khách hàng mới
#     """
#     # Lấy dữ liệu từ request
#     data = request.get_json(force=True)
#     print(data)
#     # Chuyển đổi dữ liệu thành DataFrame
#     new_customers = pd.DataFrame(data)
    
#     # Bước tiền xử lý dữ liệu để tạo ra các cột mới
#     new_customers['income_per_age'] = new_customers['income'] / new_customers['age']
#     new_customers['loan_to_income_ratio'] = new_customers['loan_amount'] / new_customers['income']
    
#     # Dự đoán A-Score
#     a_scores = model.predict_proba(new_customers)[:, 1]
    
#     # Trả về kết quả dự đoán
#     return jsonify([round(float(score), 2) for score in a_scores])

# if __name__ == '__main__':
#     app.run(debug=True)



# =============================================================================
# # Giải thích vì sao không gọi lại các bước tiền xử lý dữ liệu    
# # Không cần phải làm lại các bước tiền xử lý dữ liệu cho khách hàng mới vì đã tích hợp các bước tiền xử lý vào pipeline của mô hình. Khi gọi model.predict_proba(new_customer), pipeline sẽ tự động áp dụng các bước tiền xử lý trước khi thực hiện dự đoán. Cụ thể, pipeline của bạn bao gồm các bước sau:
# #   1. Preprocessor: Thực hiện các bước tiền xử lý dữ liệu như xử lý giá trị thiếu, mã hóa các biến phân loại và chuẩn hóa các biến số.
# #   2. Classifier: Áp dụng mô hình XGBoost để dự đoán.
# # Khi huấn luyện mô hình với model.fit(X_train, y_train), pipeline đã học cách xử lý dữ liệu từ tập huấn luyện. Do đó, khi bạn cung cấp dữ liệu mới (new_customer), pipeline sẽ tự động áp dụng các bước tiền xử lý đã học trước khi đưa dữ liệu vào mô hình để dự đoán.
# # Dưới đây là một ví dụ minh họa cách pipeline hoạt động:
# #   # Bước 1: Tiền xử lý dữ liệu
# # # Pipeline sẽ tự động thực hiện các bước tiền xử lý như xử lý giá trị thiếu, mã hóa và chuẩn hóa
# 
# # # Bước 2: Dự đoán
# # a_score = model.predict_proba(new_customer)[:, 1]
# 
# # # In kết quả A-Score
# # print(f'A-Score: {a_score[0]:.2f}')
# 
# # Pipeline giúp đơn giản hóa quy trình và đảm bảo rằng các bước tiền xử lý được áp dụng nhất quán cho cả dữ liệu huấn luyện và dữ liệu mới. Điều này giúp tránh lỗi và đảm bảo tính chính xác của dự đoán.
# 
# =============================================================================

# =============================================================================
# 
# Để test mô hình của bạn bằng Postman, bạn cần thực hiện các bước sau:
#  1. Chạy ứng dụng Flask:
#  ◦ Đảm bảo rằng bạn đã lưu file app.py và mô hình credit_model.pkl trong cùng một thư mục.
#  ◦ Mở terminal hoặc command prompt và điều hướng đến thư mục chứa file app.py.
#  ◦ Chạy lệnh sau để khởi động ứng dụng Flask:  python app.py
#  ◦ Ứng dụng Flask sẽ chạy trên http://127.0.0.1:5000/ (hoặc http://localhost:5000/).
#  2. Mở Postman:
#  ◦ Mở ứng dụng Postman trên máy tính của bạn.
#  3. Tạo một yêu cầu POST:
#  ◦ Chọn phương thức POST.
#  ◦ Nhập URL: http://127.0.0.1:5000/predict.
#  4. Thiết lập Body của yêu cầu:
#  ◦ Chọn tab Body.
#  ◦ Chọn raw và định dạng là JSON.
#  ◦ Nhập dữ liệu JSON mà bạn muốn gửi để dự đoán. Ví dụ:
#      [
#       {
#         "age": 30,
#         "income": 50000,
#         "loan_amount": 20000,
#         "gender": "male",
#         "occupation": "engineer",
#         "person_home_ownership": "rent",
#         "person_emp_length": 5,
#         "loan_intent": "education",
#         "loan_int_rate": 10.5,
#         "loan_percent_income": 0.4,
#         "cb_person_default_on_file": "N",
#         "cb_person_cred_hist_length": 10,
#         "term": "36 months",
#         "int_rate": 10.5,
#         "installment": 555.55,
#         "grade": "A",
#         "member_id": 1
#       }
#     ]
#  5. Gửi yêu cầu và kiểm tra phản hồi:
#  ◦ Nhấn nút Send để gửi yêu cầu.
#  ◦ Kiểm tra phản hồi từ server trong tab Body của Postman. Bạn sẽ nhận được kết quả dự đoán A-Score dưới dạng JSON.
# 
# =============================================================================
