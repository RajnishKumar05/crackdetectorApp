import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

st.set_page_config(page_title="Crack Detection AI", layout="wide", page_icon="🧠")

# Sidebar
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["🔍 Main App", "📏 PCI Estimator", "🧮 Cost Estimator", "📈 Model Analysis", "📜 History", "🧾 Export Report"])

# Load crack detection model
model_url = "https://huggingface.co/rj2537580/crack_detection/resolve/main/crack_detection_model.h5"
model_path = tf.keras.utils.get_file("crack_detection_model.h5", model_url)
model = tf.keras.models.load_model(model_path)
class_mapping = ['AlligatorCrack', 'LongitudinalCrack', 'TransverseCrack']

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_image" not in st.session_state:
    st.session_state.last_image = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# Helpers
def preprocess_image(image_input):
    image = Image.open(image_input).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0), image

def display_rounded_image(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(f"""
    <style>
    .rounded-image-container {{ display: flex; justify-content: flex-end; margin-top: 20px; margin-right: 40px; }}
    .rounded-image-frame {{ position: relative; width: 360px; border-radius: 24px; overflow: hidden; box-shadow: 0 16px 32px rgba(0, 0, 0, 0.25); }}
    .rounded-image-frame img {{ width: 100%; display: block; object-fit: cover; border-radius: 24px; }}
    </style>
    <div class="rounded-image-container">
        <div class="rounded-image-frame">
            <img src='data:image/png;base64,{encoded}' />
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_severity(confidence):
    if confidence >= 0.90:
        return "High"
    elif confidence >= 0.75:
        return "Medium"
    else:
        return "Low"



# Main App
if page == "🔍 Main App":
    st.title("🧠 Crack Type & Severity Detection")
    uploaded_file = st.file_uploader("📄 Upload a crack image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        processed_image, image_pil = preprocess_image(uploaded_file)
        display_rounded_image(image_pil)
        st.session_state.last_image = image_pil

        with st.spinner("🌀 Scanning image for cracks..."):
            prediction = model.predict(processed_image)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        label = class_mapping[predicted_class]
        severity = get_severity(confidence)

        st.session_state.last_prediction = {"label": label, "confidence": confidence, "severity": severity}

        st.success(f"🧠 **Predicted Class:** {label}")
        st.info(f"🎯 **Confidence:** {confidence:.2%}")
        st.warning(f"🔥 **Severity Level:** {severity}")

        st.subheader("📊 Confidence per Class")
        fig, ax = plt.subplots()
        sns.barplot(x=class_mapping, y=prediction[0], palette="coolwarm", ax=ax)
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        if st.button("💡 Give Tip to Fix It"):
            st.subheader("🤖 LLaMA 4 Scout Tip: How to Fix & Avoid This Crack")
            ask_llama3_for_solution(label, confidence, severity, st.session_state.last_image)

# PCI Estimator
elif page == "📏 PCI Estimator":
    st.title("📏 Pavement Condition Index Estimator")
    if st.session_state.last_prediction is None:
        st.info("🔍 Please scan a crack image first in the Main App.")
    else:
        label = st.session_state.last_prediction["label"]
        severity = st.session_state.last_prediction["severity"]
        penalty_scores = {
            "AlligatorCrack": {"Low": 10, "Medium": 25, "High": 40},
            "LongitudinalCrack": {"Low": 5, "Medium": 15, "High": 25},
            "TransverseCrack": {"Low": 5, "Medium": 10, "High": 20}
        }
        base_pci = 100
        penalty = penalty_scores.get(label, {}).get(severity, 0)
        estimated_pci = max(0, base_pci - penalty)
        st.write(f"🚣️ Detected Crack Type: **{label}**")
        st.write(f"🚨 Severity: **{severity}**")
        st.write(f"📉 Penalty Score: **{penalty}**")
        st.success(f"📊 Estimated PCI: **{estimated_pci}/100**")
        st.progress(estimated_pci / 100)

# Cost Estimator
elif page == "🧮 Cost Estimator":
    st.title("🧮 Crack Repair Cost Estimator")
    if st.session_state.last_prediction is None:
        st.info("📷 Analyze an image in the Main App first.")
    else:
        label = st.session_state.last_prediction["label"]
        severity = st.session_state.last_prediction["severity"]
        cost_map = {
            "AlligatorCrack": {"Low": 3000, "Medium": 4000, "High": 5000},
            "LongitudinalCrack": {"Low": 1500, "Medium": 2500, "High": 3500},
            "TransverseCrack": {"Low": 1000, "Medium": 1800, "High": 2800}
        }
        st.write(f"💪 Detected: **{label}**, Severity: **{severity}**")
        area = st.number_input("Enter Affected Area (in sq. meters):", min_value=1.0, step=0.5)
        if area:
            unit_cost = cost_map.get(label, {}).get(severity, 0)
            total_cost = unit_cost * area
            st.success(f"💰 Estimated Repair Cost: ₹{int(total_cost):,}")
            st.write(f"🧾 Unit Cost: ₹{unit_cost} per sqm")

# Model Analysis
elif page == "📈 Model Analysis":
    st.title("📈 Model Performance Simulation")
    y_true = np.array([0.95, 0.85, 0.60, 0.75, 0.92, 0.70])
    y_pred = np.array([0.93, 0.83, 0.58, 0.73, 0.91, 0.69])
    reg = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_pred, y_true, color='blue', label="Data Points")
    ax2.plot(y_pred, reg.predict(y_pred.reshape(-1, 1)), color='red', label="Regression Line")
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("True Confidence")
    ax2.set_title("Linear Regression - Confidence vs Ground Truth")
    ax2.legend()
    st.pyplot(fig2)

# History
elif page == "📜 History":
    st.title("📜 LLaMA 4 Scout Chat History")
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
    else:
        st.info("No auto-generated tips yet. Upload an image to get started!")

# Export Report
elif page == "🧾 Export Report":
    st.title("🧾 Export Crack Analysis Report")
    if st.session_state.last_prediction is None or st.session_state.last_image is None:
        st.info("📷 Analyze an image in Main App first.")
    else:
        if st.button("📅 Generate PDF Report"):
            pred = st.session_state.last_prediction
            image = st.session_state.last_image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                c = canvas.Canvas(tmpfile.name, pagesize=A4)
                width, height = A4
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, "Crack Detection Report")
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 100, f"Detected Class: {pred['label']}")
                c.drawString(50, height - 120, f"Confidence: {pred['confidence']:.2%}")
                c.drawString(50, height - 140, f"Severity: {pred['severity']}")
                img_path = os.path.join(tempfile.gettempdir(), "report_img.jpg")
                image.save(img_path)
                c.drawImage(img_path, 50, height - 350, width=300, preserveAspectRatio=True)
                if st.session_state.chat_history:
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(50, height - 380, "AI Suggestions:")
                    text = st.session_state.chat_history[-1]["content"]
                    text_lines = text.split("\n")
                    y = height - 400
                    for line in text_lines:
                        c.drawString(50, y, line[:90])
                        y -= 15
                        if y < 50:
                            c.showPage()
                            y = height - 50
                c.save()
                with open(tmpfile.name, "rb") as f:
                    st.download_button(
                        label="📅 Download Report PDF",
                        data=f.read(),
                        file_name="crack_analysis_report.pdf",
                        mime="application/pdf"
                    )