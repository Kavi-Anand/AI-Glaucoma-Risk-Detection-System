import os
os.environ["KERAS_BACKEND"] = "torch"
import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import pennylane as qml
import keras
from keras.models import load_model
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Glaucoma Risk Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        background: white;
    }
    .risk-low { color: #059669; font-weight: 800; font-size: 1.8rem; }
    .risk-moderate { color: #D97706; font-weight: 800; font-size: 1.8rem; }
    .risk-high { color: #DC2626; font-weight: 800; font-size: 1.8rem; }
    .metric-label { font-size: 1.1rem; color: #4B5563; font-weight: 600; }
    .metric-value { font-size: 1.5rem; color: #111827; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# --- QUANTUM ARCHITECTURE DEFINITION ---
n_qubits = 8
try:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        # Encoding: RX(x*pi), RZ(x*pi) 
        for i in range(n_qubits):
            qml.RX(inputs[i] * torch.pi, wires=i)
            qml.RZ(inputs[i] * torch.pi, wires=i)
            
        # RY(weights)
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
            
        # CNOT chain
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        
        # RX(weights)
        for i in range(n_qubits):
            qml.RX(weights[n_qubits + i], wires=i)
            
        # Output: PauliZ(0)
        return qml.expval(qml.PauliZ(0))

    class QuantumNet(nn.Module):
        def __init__(self):
            super(QuantumNet, self).__init__()
            self.weights = nn.Parameter(torch.randn(16))
            self.scale = nn.Parameter(torch.tensor(1.0))
            
        def forward(self, x):
            logits = torch.stack([quantum_circuit(xi, self.weights) for xi in x])
            return logits * self.scale
except Exception as e:
    st.error(f"Error initializing quantum simulation backend: {e}")


# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_quantum_model(model_path="quantum_model_g1020.pth"):
    if not os.path.exists(model_path):
        return None
    try:
        loaded = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(loaded, dict):
            # Extract the correct dictionary
            state_dict = loaded.get('model_state_dict', loaded)
            model = QuantumNet()
            model.load_state_dict(state_dict)
        else:
            model = loaded
        model.eval()
        return model
    except Exception as e:
        import traceback
        st.error(f"Error loading Quantum model: {traceback.format_exc()}")
        return None

@st.cache_resource
def load_classical_model(model_path="glaucoma_mobilenet_final.keras"):
    if not os.path.exists(model_path):
        return None
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading Classical model: {e}")
        return None


# --- PREPROCESSING FUNCTIONS ---
def preprocess_quantum(image: Image.Image):
    """
    Quantum Preprocessing:
    - Grayscale
    - Resize 16x16
    - Normalize [0,1]
    - Flatten
    - Split into 8 chunks -> mean -> 8 features
    """
    img = image.convert("L")
    img = img.resize((16, 16))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_flat = img_array.flatten()
    
    # 256 / 8 = 32 items per chunk
    lengths = len(img_flat) // 8
    features = [np.mean(img_flat[i*lengths : (i+1)*lengths]) for i in range(8)]
    return torch.tensor([features], dtype=torch.float32)

def preprocess_classical(image: Image.Image):
    """
    Classical Preprocessing:
    - RGB
    - Resize (224, 224)
    - Normalize by dividing by 255
    - NO preprocess_input
    """
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# --- PLOTLY GAUGE GENERATOR ---
def create_gauge_chart(probability, threshold, high_threshold):
    # Determine color of the thick bar based on probability
    if probability < threshold:
        bar_color = "#059669" # green
    elif probability < high_threshold:
        bar_color = "#D97706" # yellow
    else:
        bar_color = "#DC2626" # red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Glaucoma Probability (%)", 'font': {'size': 20, 'color': '#111827'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': bar_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color, 'thickness': 0.25}, # This acts as our "needle" or indicator bar
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': "rgba(5, 150, 105, 0.15)"},
                {'range': [threshold * 100, high_threshold * 100], 'color': "rgba(217, 119, 6, 0.15)"},
                {'range': [high_threshold * 100, 100], 'color': "rgba(220, 38, 38, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        font={'family': "Inter, sans-serif"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# --- MAIN APP ROUTING ---
def main():
    st.markdown("<div class='main-header'>AI Glaucoma Risk Detection System</div>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        st.markdown("---")
        model_choice = st.radio("Select Prediction Engine:", [
            "Quantum Model", 
            "Classical Model"
        ], index=0)
        
        st.markdown("---")
        st.markdown("**Thresholds in Use:**")
        if model_choice == "Quantum Model":
            st.code("Low Risk: < 45%\nMod Risk: 45%-70%\nHigh Risk: > 70%")
        else:
            st.code("Low Risk: < 30%\nMod Risk: 30%-60%\nHigh Risk: > 60%")
            
        st.markdown("---")
        st.caption("Place `quantum_model_g1020.pth` and `glaucoma_mobilenet_final.keras` in the same directory.")
        
    st.markdown("### 1. Upload Retinal Scan")
    uploaded_file = st.file_uploader("Upload a Retinal Fundus Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 2. Analysis Results")
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1.3])
        
        with col1:
            st.image(image, caption="Uploaded Fundus Image", use_container_width=True)
            
        with col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            with st.spinner(f"Processing with {model_choice}..."):
                probability = None
                
                # Inference
                if model_choice == "Quantum Model":
                    model = load_quantum_model()
                    if model is not None:
                        input_tensor = preprocess_quantum(image)
                        with torch.no_grad():
                            logits = model(input_tensor)
                            if logits.dim() == 0 or logits.numel() == 1:
                                prob = torch.sigmoid(logits).item()
                            else:
                                prob = torch.sigmoid(logits).squeeze().item()
                            probability = prob
                        
                        threshold = 0.45
                        high_thresh = 0.60
                    else:
                        st.error("Quantum model file (`quantum_model_g1020.pth`) is missing. Please ensure it is placed in the project directory.")
                else:
                    model = load_classical_model()
                    if model is not None:
                        input_tensor = preprocess_classical(image)
                        preds = model.predict(input_tensor)
                        probability = float(preds[0][1])
                        
                        threshold = 0.30
                        high_thresh = 0.50
                    else:
                        st.error("Classical model file (`glaucoma_mobilenet_final.keras`) is missing. Please ensure it is placed in the project directory.")
                
                # Render Results
                if probability is not None:
                    # Risk Classification Logic
                    if probability < threshold:
                        risk_label = "LOW RISK"
                        risk_class = "risk-low"
                    elif probability < high_thresh:
                        risk_label = "MODERATE RISK"
                        risk_class = "risk-moderate"
                    else:
                        risk_label = "HIGH RISK"
                        risk_class = "risk-high"
                        
                    # Confidence Calculation 
                    if probability < threshold:
                        confidence = 0.5 + 0.5 * ((threshold - probability) / threshold)
                    else:
                        if threshold < 1.0:
                            confidence = 0.5 + 0.5 * ((probability - threshold) / (1.0 - threshold))
                        else:
                            confidence = 1.0
                            
                    confidence_pct = confidence * 100
                    
                    st.markdown(f"<div style='text-align:center;'>Classified as:<br> <span class='{risk_class}'>{risk_label}</span></div>", unsafe_allow_html=True)
                    
                    # Layout inner columns for metrics
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"<div class='metric-label'>Prob (Glaucoma)</div><div class='metric-value'>{probability * 100:.1f}%</div>", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"<div class='metric-label'>Confidence</div><div class='metric-value'>{confidence_pct:.1f}%</div>", unsafe_allow_html=True)
                    
                    # Plotly chart
                    fig = create_gauge_chart(probability, threshold, high_thresh)
                    st.plotly_chart(fig, use_container_width=True)
                    
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 3. Model Comparison")
        with st.spinner("Generating comparison..."):
            prob_quantum = None
            prob_classical = None
            
            # Quantum Model Probability
            if model_choice == "Quantum Model" and probability is not None:
                prob_quantum = probability
            else:
                q_model = load_quantum_model()
                if q_model is not None:
                    q_input = preprocess_quantum(image)
                    with torch.no_grad():
                        logits = q_model(q_input)
                        if logits.dim() == 0 or logits.numel() == 1:
                            prob_quantum = torch.sigmoid(logits).item()
                        else:
                            prob_quantum = torch.sigmoid(logits).squeeze().item()

            # Classical Model Probability
            if model_choice == "Classical Model" and probability is not None:
                prob_classical = probability
            else:
                c_model = load_classical_model()
                if c_model is not None:
                    c_input = preprocess_classical(image)
                    preds = c_model.predict(c_input)
                    prob_classical = float(preds[0][1])
            
            # Render Comparison Chart
            if prob_quantum is not None and prob_classical is not None:
                fig_comp = go.Figure(data=[
                    go.Bar(
                        x=['Quantum Model', 'Classical Model'], 
                        y=[prob_quantum * 100, prob_classical * 100], 
                        marker_color=['#1E3A8A', '#059669'],
                        text=[f"{prob_quantum * 100:.1f}%", f"{prob_classical * 100:.1f}%"],
                        textposition='auto',
                        width=[0.4, 0.4]
                    )
                ])
                fig_comp.update_layout(
                    title='Glaucoma Probability by Model',
                    yaxis_title='Probability (%)',
                    yaxis=dict(range=[0, 100]),
                    font=dict(family="Inter, sans-serif"),
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Both models must be available to display the comparison chart.")

if __name__ == "__main__":
    main()
