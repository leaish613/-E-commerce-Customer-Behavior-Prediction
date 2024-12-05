
# E-commerce Customer Behavior Prediction

## Overview

This project involves building a **machine learning model** to predict **customer satisfaction** based on various features such as **Age**, **Gender**, **City**, **Membership Type**, and **Total Spend** using the **Random Forest Classifier**. The dataset used for this project focuses on **customer engagement** and **purchasing patterns** from an e-commerce platform.

The goal is to predict whether a customer will be **satisfied**, **neutral**, or **unsatisfied** based on historical data, which can then be used to make informed decisions regarding marketing and customer service.

## Dataset

The dataset contains the following columns:
- **Customer ID**: Unique identifier for each customer.
- **Gender**: Gender of the customer (e.g., Male, Female).
- **Age**: Age of the customer.
- **City**: The city where the customer resides.
- **Membership Type**: Customer's membership tier (e.g., Gold, Silver, Bronze).
- **Total Spend**: Total amount spent by the customer.
- **Items Purchased**: Number of items purchased by the customer.
- **Average Rating**: Average rating given by the customer.
- **Discount Applied**: Whether a discount was applied to the purchase (True/False).
- **Days Since Last Purchase**: Number of days since the customer made their last purchase.
- **Satisfaction Level**: Customer’s satisfaction level (e.g., Satisfied, Neutral, Unsatisfied).

The dataset can be loaded from `E-commerce Customer Behavior - Sheet1.csv`.

## Installation

To run this project, you need Python along with the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required libraries by running:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Steps

### 1. **Data Preprocessing**
   - **Handle missing values**: Missing values are filled with the mode for categorical variables and mean/median for numerical variables.
   - **Encode categorical variables**: Label encoding is used to convert categorical data (e.g., Gender, City, Satisfaction Level) into numerical format.

### 2. **Model Training**
   - **Train-Test Split**: The dataset is split into training (80%) and testing (20%) sets.
   - **Model**: A **Random Forest Classifier** is used to train the model and predict customer satisfaction levels.

### 3. **Model Evaluation**
   - **Accuracy**: The model’s accuracy is evaluated based on the test set.
   - **Confusion Matrix**: A confusion matrix is plotted to visualize model performance.
   - **Feature Importance**: The importance of each feature in predicting customer satisfaction is displayed using a bar plot.

### 4. **Future Improvements**
   - **Hyperparameter Tuning**: Using GridSearchCV or RandomizedSearchCV to find the best parameters for the model.
   - **Other Models**: Exploring other machine learning models such as **Logistic Regression**, **Support Vector Machines**, or **Gradient Boosting**.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/e-commerce-customer-behavior-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd e-commerce-customer-behavior-prediction
   ```

3. Run the project in a Python environment:
   ```bash
   python customer_satisfaction_prediction.py
   ```

## Results

- **Accuracy Score**: Provides the percentage of correctly classified instances from the test data.
- **Confusion Matrix**: Helps evaluate how well the model performs in predicting different satisfaction levels (Satisfied, Neutral, Unsatisfied).
- **Feature Importance**: Displays which features most affect the predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset is taken from a publicly available e-commerce customer behavior dataset.
- Libraries used: **pandas**, **numpy**, **scikit-learn**, **matplotlib**, **seaborn**.
