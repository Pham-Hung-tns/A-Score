# Overview:
- ## Objectives:
		- Build an A-Score (Application Score) predictive model using Python and the XGBoost algorithm.
		- Deploy the model via the Flask API.
- ## Data preparation:
		- Read data from a CSV file.
		- Handle missing values ​​using SimpleImputer (median for numerical variables and most common value for categorical variables).
		- Create new variables from existing data such as income_per_age and loan_to_income_ratio.
- ## Model building:
		- Use XGBoost to build a predictive model.
		- Evaluate the model using accuracy and other metrics.
- ## Build Flask API and test model with Postman
# Project launch guide (with Postman)
1. **Download Postman and set up an account**: [https://www.postman.com/](https://www.postman.com/)
2. **Launch the project (VSCode environment)**
![Launch the project with the default port: http://127.0.0.1:5000](https://github.com/user-attachments/assets/05c68ec6-7b05-4d33-9059-b49617fc9710)
3. **Set up in Postman**
- Create a new Collection (**A-Score**)
- In the Method section, select **POST**, url to test: *http://127.0.0.1:5000/predict*
- With the data to test, you can create new ones yourself according to structure below:
 ```
 [{
 "age": 30,
 "income": 50000,
 "loan_amount": 20000,
 "gender": "male",
 "occupation": "engineer",
 "person_home_ownership": "rent",
 "person_emp_length": 5,
 "loan_intent": "education",
 "loan_int_rate": 10.5,
 "loan_status": 0,
 "loan_percent_income": 0.4,
 "cb_person_default_on_file": "N",
 "cb_person_cred_hist_length": 10,
 "term": "36 months",
 "int_rate": 10.5,
 "installation": 555.55,
 "grade": "A",
 "member_id": 1
}]
```
# Development(In the future):
- Create a web interface for users to enter values ​​and return prediction results.
- Save data to database
- Add, adjust other parameters to the model (**n_estimators**, **max_depth**, **learning_rate**,...) to get better results!
