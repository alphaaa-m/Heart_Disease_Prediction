# Heart Disease Prediction Using Machine Learning

![Heart_Disease_Cover](https://github.com/alphaaa-m/Heart_Disease_Prediction/raw/main/Heart_Disease_Cover.png)
## 📋 Project Overview

Heart disease is one of the leading causes of death globally. Early detection can significantly improve treatment outcomes and save lives.

In this project, we use **clinical and demographic attributes** (like age, blood pressure, cholesterol levels, chest pain type, etc.) to predict the presence of heart disease using machine learning models.

We have built and evaluated multiple machine learning models, aiming to find the most accurate and reliable one for heart disease prediction.

---

## 📂 Project Structure

- `Heart_Disease.ipynb`: Jupyter Notebook containing full code — from data preprocessing to model evaluation.
- `Heart_Disease_Cover.pdf`: Project cover image.

---

## ⚙️ Steps Followed

1. **Data Preprocessing**:
   - Handled missing values (especially in the `Thalassemia` column).
   - Applied **OneHotEncoding** on categorical features.
   - Feature scaling using **StandardScaler**.

2. **Exploratory Data Analysis (EDA)**:
   - Used correlation heatmaps to understand relationships between features.

3. **Model Building and Evaluation**:
   - Built and trained four models:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Decision Tree Classifier
     - Random Forest Classifier
   - Evaluated models using:
     - Accuracy
     - Confusion Matrices
     - Precision, Recall, F1-Score reports

4. **Model Comparison**:
   | Model | Accuracy |
   |:-----:|:--------:|
   | Logistic Regression | 81% |
   | Support Vector Machine (SVM) | 86% |
   | Decision Tree Classifier | 82% |
   | Random Forest Classifier | **89%** |

5. **Conclusion**:
   - **Random Forest Classifier** performed best with **89% accuracy**.
   - Future improvements suggested include hyperparameter tuning (e.g., GridSearchCV), cross-validation, and trying boosting algorithms like XGBoost.

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/alphaaa-m/Heart_Disease_Prediction.git
   cd Heart_Disease_Prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually install packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)*

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Heart_Disease.ipynb
   ```

4. Execute all the cells and view the results.

---

## 🧠 Key Libraries Used
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## 👨‍💻 Author

**Muneeb Ashraf**

---


---

## 📢 Connect with Me
Feel free to connect for collaborations, feedback, or suggestions!

---
> **GitHub Repo Link**: [Visit Here](https://github.com/alphaaa-m/Heart-Disease-Prediction)
