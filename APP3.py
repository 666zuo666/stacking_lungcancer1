import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# 加载模型
model_path = "stacking_classifier_model.pkl"
stacking_classifier = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")

st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# 左侧侧边栏输入区域
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
ITH_score = st.number_input("ITH score:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
Size = st.number_input("Tumor size:", min_value=0.0, max_value=30.0, value=7.62, step=0.01)
Mean_CT_value = st.number_input("Mean CT value:", min_value=-800.0, max_value=0.0, value=-480.66, step=1.0)
Pleural_indentation = st.selectbox("Pleural indentation:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Age = st.number_input("Age:", min_value=21.0, max_value=100.0, value=64.0, step=1.0)
Location = st.selectbox("Location:", options=[1, 2, 3, 4, 5], format_func=lambda x: "RUL" if x == 1 else ("RLL" if x == 2 else ("RML" if x == 3 else ("LUL" if x == 4 else "LLL"))))
Shape = st.selectbox("Shape:", options=[0, 1], format_func=lambda x: "Regular" if x == 1 else "Irregular")
Vacuole_sign = st.selectbox("Vacuole_sign:", options=[0, 1], format_func=lambda x: "absent" if x == 1 else "present")
Sex = st.selectbox("Sex:", options=[0, 1], format_func=lambda x: "absent" if x == 1 else "present")
Margin = st.selectbox("Margin:", options=[0, 1], format_func=lambda x: "clear" if x == 1 else "unclear")
Spiculation = st.selectbox("Spiculation:", options=[0, 1], format_func=lambda x: "absent" if x == 1 else "present")
Lobulation = st.selectbox("Lobulation:", options=[0, 1], format_func=lambda x: "absent" if x == 1 else "present")
Vascular_convergence_sign = st.selectbox("Vascular_convergence_sign:", options=[0, 1], format_func=lambda x: "absent" if x == 1 else "present")
# 添加预测按钮
predict_button = st.sidebar.button("Predict")

# 主页面用于结果展示
if predict_button:
    st.header("Predict Result")
    try:
        # # 将输入特征转换为模型所需格式
        # input_array = np.array([X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8]).reshape(1, -1)
        # 处理输入并进行预测
        feature_values = [ITH_score, Size, Mean_CT_value, Pleural_indentation, Age, Location, Shape, Spiculation, Margin, Sex, Vacuole_sign, Vascular_convergence_sign, Lobulation]
        input_array = np.array([feature_values])
        # 模型预测
        prediction = stacking_classifier.predict(input_array)[0]

        # 显示预测结果
        st.success(f"Predict Result：{prediction:.2f}")
    except Exception as e:
        st.error(f"Error：{e}")

# 可视化展示
st.header("SHAP")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括第一层基学习器、第二层元学习器以及整个 Stacking 模型的特征贡献。
""")

# 第一层基学习器 SHAP 可视化
st.subheader("1. Base_learners")
st.write("基学习器（RandomForest、XGB、LGBM 等）的特征贡献分析。")
first_layer_img = "summary_plot.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="第一层基学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第一层基学习器的 SHAP 图像文件。")

# 第二层元学习器 SHAP 可视化
st.subheader("2. Stacking_classifier")
st.write("元学习器（Linear Regression）的输入特征贡献分析。")
meta_layer_img = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="第二层元学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第二层元学习器的 SHAP 图像文件。")

# 整体 Stacking 模型 SHAP 可视化
st.subheader("3. Stacking")
st.write("整个 Stacking 模型的特征贡献分析。")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="整体 Stacking 模型的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

# 页脚
st.markdown("---")
st.header("总结")
st.write("""
通过本页面，您可以：
1. 使用输入特征值进行实时预测。
2. 直观地理解第一层基学习器、第二层元学习器以及整体 Stacking 模型的特征贡献情况。
这些分析有助于深入理解模型的预测逻辑和特征的重要性。
""")
