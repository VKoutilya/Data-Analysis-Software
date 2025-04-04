import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import webbrowser
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

class RegressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.file_path = ""

    def initUI(self):
        layout = QVBoxLayout()
        
        self.label = QLabel("Select a CSV or Excel file")
        layout.addWidget(self.label)
        
        self.btn_file = QPushButton("Load File")
        self.btn_file.clicked.connect(self.load_file)
        layout.addWidget(self.btn_file)
        
        self.degree_input = QLineEdit(self)
        self.degree_input.setPlaceholderText("Enter max polynomial degree (1-10)")
        layout.addWidget(self.degree_input)
        
        self.btn_analyze = QPushButton("Find Best Fit")
        self.btn_analyze.clicked.connect(self.compare_best_fit)
        layout.addWidget(self.btn_analyze)
        
        self.setLayout(layout)
        self.setWindowTitle("Regression Analysis")
        self.setGeometry(100, 100, 400, 200)

    def load_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)", options=options)
        if file_name:
            self.file_path = file_name
            self.label.setText(f"Selected: {self.file_path}")
    
    def compare_best_fit(self):
        if not self.file_path:
            self.show_message("Error", "No file selected")
            return
        
        try:
            if self.file_path.endswith(".csv"):
                df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(".xlsx"):
                df = pd.read_excel(self.file_path)
            else:
                self.show_message("Error", "Unsupported file format")
                return
            
            if len(df.columns) < 2:
                self.show_message("Error", "Dataset must have at least two columns (X and Y)")
                return
            
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            results = {}
            
            results["Linear"] = self.linear_regression(X, y)
            results["Multiple Linear"] = self.multiple_linear_regression(X, y)
            
            try:
                max_degree = self.degree_input.text().strip()
                if not max_degree.isdigit():
                    self.show_message("Error", "Invalid polynomial degree")
                    return
                max_degree = int(max_degree)
                if max_degree < 1 or max_degree > 10:
                    self.show_message("Error", "Degree must be between 1 and 10")
                    return
                
                for degree in range(1, max_degree + 1):
                    results[f"Polynomial (Degree {degree})"] = self.polynomial_regression(X, y, degree)
            except ValueError:
                self.show_message("Error", "Invalid polynomial degree")
                return
            
            results = {k: v for k, v in results.items() if v}
            best_model = max(results.items(), key=lambda x: x[1]['r2'])
            best_name, best_result = best_model
            self.plot_comparison_graph(results, best_name)
            self.show_message("Best Fit", f"The best fit is {best_name} Regression (R² = {best_result['r2']:.4f})")
        except Exception as e:
            self.show_message("Error", f"Something went wrong: {e}")

    def linear_regression(self, X, y):
        if X.shape[1] > 1:
            return None
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return {"X": X, "y": y, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    def multiple_linear_regression(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return {"X": X[:, 0], "y": y, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    def polynomial_regression(self, X, y, degree):
        if X.shape[1] > 1:
            return None
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        return {"X": X, "y": y, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    def plot_comparison_graph(self, results, best_fit_name):
        fig = go.Figure()
        colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
        
        for i, (name, result) in enumerate(results.items()):
            X, y, y_pred, r2 = result.values()
            color = colors[i % len(colors)]
            width = 3 if name == best_fit_name else 1
            fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='lines', name=f"{name} (R²={r2:.4f})", line=dict(color=color, width=width)))
        
        fig.add_trace(go.Scatter(x=results[best_fit_name]["X"].flatten(), y=results[best_fit_name]["y"], mode='markers', name="Actual Data", marker=dict(color="black")))
        fig.update_layout(title="Regression Model Comparison", xaxis_title="X", yaxis_title="Y")
        fig.write_html("regression_comparison.html")
        webbrowser.open("regression_comparison.html")

    def show_message(self, title, message):
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RegressionApp()
    window.show()
    sys.exit(app.exec_())
