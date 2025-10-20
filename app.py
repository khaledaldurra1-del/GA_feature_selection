import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ إعداد واجهة Streamlit
# -------------------------------
st.set_page_config(page_title="GA Feature Selection", layout="wide")

st.title("Genetic Algorithm Feature Selection")
st.write("Upload your dataset and see the best features selected by GA.")

# -------------------------------
# 2️⃣ رفع الملف
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("First 5 rows of the dataset")
    st.dataframe(data.head())
    
    st.subheader("Columns in the dataset")
    st.write(list(data.columns))
    
    # اختيار عمود الهدف
    target_column = st.selectbox("Select the target column", options=data.columns)
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # -------------------------------
    # 3️⃣ استعراض نتائج GA
    # -------------------------------
    st.subheader("GA Results (previously computed)")
    
    try:
        results_df = pd.read_csv("selected_features_results.csv")
        st.write("Selected Features:")
        st.dataframe(results_df)
    except FileNotFoundError:
        st.warning("No GA results found. Run ga_demo.py first to generate 'selected_features_results.csv'.")
    
    # -------------------------------
    # 4️⃣ عرض الرسم البياني لتطور Fitness
    # -------------------------------
    try:
        fitness_plot = "fitness_plot.png"
        st.subheader("Fitness Evolution")
        st.image(fitness_plot, caption="Fitness over generations", use_column_width=True)
    except FileNotFoundError:
        st.warning("No fitness plot found. Run ga_demo.py first to generate 'fitness_plot.png'.")
    
    # -------------------------------
    # 5️⃣ مقارنة الأداء بين الطريقتين
    # -------------------------------
    try:
        comparison_plot = "comparison_accuracy.png"
        st.subheader("Comparison: All Features vs GA Features")
        st.image(comparison_plot, caption="Accuracy Comparison", use_column_width=True)
    except FileNotFoundError:
        st.warning("No comparison plot found. Run ga_demo.py first to generate 'comparison_accuracy.png'.")
