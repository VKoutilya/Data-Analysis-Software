import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_data(file):
    """Load data and remove columns with all NaN values."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Drop fully empty columns and keep only numeric data
    df = df.dropna(axis=1, how='all').select_dtypes(include=[np.number])  
    return df

def clean_data(X, y):
    """Drop rows where X or y contain NaN values."""
    data = np.hstack((X, y.reshape(-1, 1)))  # Combine X and y into one array
    clean_data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN
    return clean_data[:, :-1], clean_data[:, -1]  # Return cleaned X, y separately

def polynomial_regression(X, y, degrees):
    results = {}
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        equation = format_equation(model, degree)
        results[degree] = (model, y_pred, r2, equation)
    return results

def format_equation(model, degree):
    """Generate a polynomial equation in the form y = ax^n + bx^(n-1) + ... + c."""
    coefs = model.coef_.flatten()
    intercept = model.intercept_

    terms = [f"{coefs[i]:.4f}x^{i}" for i in range(1, len(coefs)) if coefs[i] != 0]
    equation = " + ".join(terms)

    # Always include the constant term "+ c"
    equation = f"y = {equation} + {intercept:.4f}" if equation else f"y = {intercept:.4f}"
    
    return equation.replace("+ -", "- ")  # Clean up negative terms

st.title("Polynomial Regression Analysis")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

degree_range = st.sidebar.slider("Polynomial Degree Range", 1, 10, (1, 5))

if uploaded_file:
    df = load_data(uploaded_file)

    if df.empty:
        st.error("No valid numeric data found. Please upload a file with numeric values.")
    else:
        columns = df.columns.tolist()

        # Searchable dropdown for selecting X and Y columns
        x_col = st.sidebar.selectbox("Select X column", columns, help="Search for a column name")
        y_col = st.sidebar.selectbox("Select Y column", columns, index=1 if len(columns) > 1 else 0, help="Search for a column name")

        X = df[[x_col]].values
        y = df[y_col].values

        # Remove NaN rows
        X, y = clean_data(X, y)

        if len(X) == 0:
            st.error("All rows contain NaN values. Please upload a file with complete numeric data.")
        else:
            degrees = list(range(degree_range[0], degree_range[1] + 1))
            results = polynomial_regression(X, y, degrees)

            # Identify best-fit model
            best_degree = max(results, key=lambda d: results[d][2])
            best_r2 = results[best_degree][2]
            best_equation = results[best_degree][3]

            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Data', marker=dict(size=6, opacity=0.7)))
            
            X_plot = np.linspace(min(X), max(X), 500).reshape(-1, 1)
            for degree, (model, _, r2, equation) in results.items():
                poly = PolynomialFeatures(degree)
                X_poly_plot = poly.fit_transform(X_plot)
                y_plot = model.predict(X_poly_plot)
                
                line_style = dict(width=4 if degree == best_degree else 2)
                fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_plot, mode='lines', 
                                         name=f"Degree {degree} (R²: {r2:.2f})", line=line_style))
            
            fig.update_layout(title="Polynomial Regression Fit", xaxis_title=x_col, yaxis_title=y_col, legend_title="Degree")
            st.plotly_chart(fig)

            # Show R² values and equations
            st.write("### Regression Equations and R² Scores")
            for degree, (_, _, r2, equation) in results.items():
                st.write(f"**Degree {degree}:** R² = {r2:.4f}")
                st.latex(equation.replace('*', '').replace('x^1', 'x'))  # Formatting equation for LaTeX display

            # Display Best Fit Model
            st.write("##  Best-Fit Model")
            st.write(f"**Best Degree:** {best_degree} (Highest R²: {best_r2:.4f})")
            st.latex(best_equation.replace('*', '').replace('x^1', 'x'))  # Formatting for LaTeX display
else:
    st.write("Please upload an Excel or CSV file containing numeric data.")
