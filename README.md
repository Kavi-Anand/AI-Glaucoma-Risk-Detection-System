# 👁️ AI Glaucoma Risk Detection System

An AI-powered glaucoma risk detection system that uses Deep Learning and Quantum Machine Learning techniques to analyze retinal images and assist in identifying potential glaucoma cases.

## 📌 Project Overview

Glaucoma is a serious eye disease that can lead to permanent vision loss if not detected early. Traditional diagnosis requires specialized examination and medical expertise.

This project aims to develop an intelligent system that analyzes retinal images using Deep Learning and Quantum Machine Learning models to classify potential glaucoma cases.

The application provides an interactive web interface developed using Streamlit, allowing users to upload retinal images and view prediction results.

## 🎯 Objectives

- Develop an AI-based system for glaucoma risk detection.
- Apply MobileNetV2 for retinal image feature extraction and classification.
- Explore Quantum Machine Learning for image classification.
- Integrate Deep Learning and Quantum Machine Learning approaches.
- Provide an interactive dashboard for image upload and prediction.
- Display prediction results through a user-friendly interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application development |
| TensorFlow / Keras | Deep Learning |
| MobileNetV2 | Retinal image classification |
| PyTorch | Quantum model integration |
| PennyLane | Quantum Machine Learning |
| NumPy | Numerical computations |
| Pillow (PIL) | Image processing |
| Plotly | Data visualization |

## 🧠 Models Used

### 1. MobileNetV2

MobileNetV2 is a lightweight convolutional neural network used for extracting meaningful features from retinal images and performing classification.

### 2. Quantum Machine Learning

A Quantum Machine Learning model is implemented using PennyLane and PyTorch to explore quantum-based classification.

### 3. Hybrid AI Approach

The project explores combining classical Deep Learning feature extraction with a quantum classification approach.

## ⚙️ Project Structure

```text
AI-Glaucoma-Risk-Detection-System/
│
├── app.py
├── glaucoma_mobilenet_final.keras
├── quantum_model_g1020.pth
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kavi-Anand/AI-Glaucoma-Risk-Detection-System.git
```

### Step 2: Navigate to the Project Folder

```bash
cd AI-Glaucoma-Risk-Detection-System
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

### Step 7: Open in Browser

```text
http://localhost:8501
```

## 💻 Application Features

- Interactive Streamlit dashboard.
- Retinal image upload.
- Image preprocessing and analysis.
- Deep Learning-based glaucoma classification.
- Quantum Machine Learning integration.
- Prediction results and visualizations.

*The availability of individual features depends on the implementation in the current application.*

## 📊 Expected Workflow

```text
User
  |
  v
Upload Retinal Image
  |
  v
Image Preprocessing
  |
  v
AI Model Processing
  |
  v
Glaucoma Classification
  |
  v
Display Prediction Results
```

## 🔮 Future Enhancements

- Improve classification accuracy using larger retinal image datasets.
- Integrate additional retinal image datasets.
- Optimize the hybrid Deep Learning and Quantum Learning architecture.
- Add explainable AI techniques such as Grad-CAM.
- Deploy the application on a cloud platform.
- Improve prediction reporting and visualization.

## ⚠️ Disclaimer

This project is developed for academic and research purposes. It is not intended to replace professional ophthalmological examination or medical diagnosis. Predictions should not be used as the sole basis for medical decisions.

## 👨‍💻 Developer

**Kavi Anand**

Computer Science and Engineering

GitHub: [Kavi-Anand](https://github.com/Kavi-Anand)

## 📜 License

This project is intended for educational and research purposes. Add an appropriate open-source license before permitting external reuse.
