# Data-Analysis-Software
This project is a user-friendly web application that enables users to perform and visualize polynomial regression analysis without writing any code. Built with Streamlit, the app allows users to upload CSV files, select variables, choose polynomial degrees, and instantly view fitted curves 
# 📊 Polynomial Regression Streamlit App

This project is a user-friendly web application for performing polynomial regression analysis without writing any code. Built using **Streamlit**, the app enables users to explore trends in their data interactively and visually by comparing polynomial models of various degrees.

---

## 🔍 Description

This tool was created to simplify the process of polynomial regression, which is commonly used in fields like engineering, business analytics, education, and research. Traditional regression software often requires coding knowledge or expensive licenses, making it difficult for non-technical users to leverage its full potential.

This app addresses that gap by offering a clean, intuitive interface that supports the entire workflow:

- Upload your own CSV dataset
- Choose any numeric columns for input (X) and output (Y)
- Select a range of polynomial degrees (e.g., 1–10)
- Instantly view regression curves and R² scores
- Automatically highlight the best-fitting model
- Download results for further use

Whether you're a student learning about regression, a researcher analyzing patterns, or an engineer building predictive models, this tool provides the clarity and speed you need — no programming required.

---

## ⚙️ Features

- ✅ Manual polynomial degree selection
- 📈 Compare multiple regression models visually
- 🏆 Best-fit model automatically highlighted using R²
- 🔄 Supports any numeric input/output columns
- 💡 Built with open-source tools for full transparency

---

## 🛠️ Requirements

- Python 3.8 or higher
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt



TO clone:
git clone https://github.com/yourusername/regression-streamlit-app.git
cd regression-streamlit-app
streamlit run app.py

