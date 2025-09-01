import streamlit as st 
import numpy as np 
import pandas as pd
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import SGD
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import EarlyStopping
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import graphviz

# =======================
# Custom CSS Styling
# =======================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #141e30, #243b55);
            font-family: 'Poppins', sans-serif;
            color: #f8f9fa;
        }
        h1, h2, h3, h4, h5 {
            color: #00d4ff;
            text-align: center;
            font-weight: 700;
        }
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }
        section[data-testid="stSidebar"] * {
            color: #f8f9fa !important;
        }
        div.stButton > button {
            background-color: #00d4ff;
            color: #141e30;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #ff4081;
            color: #fff;
        }
        .stSelectbox, .stNumberInput, .stSlider, .stTextInput {
            border-radius: 8px !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# Title & Description
# =======================
st.title("⚡ TensorFlow Playground")
st.markdown("<p style='text-align:center;font-size:18px;'>An interactive platform to visualize decision boundaries and explore neural networks.</p>", unsafe_allow_html=True)

# =======================
# Sidebar Controls
# =======================
with st.sidebar:
    st.header("📂 Dataset Settings")
    dataset = st.selectbox("Select Dataset", ["Blobs", "Circles", "Moons", "Upload CSV"])

    noise = st.slider("Noise", 0.0, 1.0, 0.1)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    st.header("⚙️ Model Hyperparameters")
    neurons_text = st.text_input("Neurons per hidden layer", placeholder="e.g. 8,16,32")
    parse_neurons = lambda x: [int(i.strip()) for i in x.split(",") if i.strip() != ""]
    nn = parse_neurons(neurons_text)

    epochs = st.number_input("Epochs", 1, 10000, step=1, value=50)

    col1, col2 = st.columns(2)
    with col1:
        af = st.selectbox("Activation Function", ["sigmoid", "tanh", "relu"], index=2)
    with col2:
        lr = st.selectbox("Learning Rate", [0.1, 0.05, 0.02, 0.01], index=3)

    reg_choice = st.selectbox("Regularizer", ["None", "L1", "L2", "ElasticNet"])
    reg_rate = None
    if reg_choice != "None":
        reg_rate = st.slider("Regularization Rate", 0.0, 0.1, 0.01)

    es = st.selectbox("Early Stopping", ["No", "Yes"], index=0)
    if es == "Yes":
        col3, col4 = st.columns(2)
        with col3:
            min_delta = st.number_input("Minimum Delta", 0.001, 0.9, step=0.01, value=0.01)
        with col4:
            patience = st.number_input("Patience", 3, 20, step=1, value=5)

# =======================
# Dataset Loading
# =======================
if dataset == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        feature_cols = st.sidebar.multiselect("Select exactly 2 features", df.columns[:-1])
        target_col = st.sidebar.selectbox("Select target column", df.columns)
        if len(feature_cols) == 2:
            X = df[feature_cols].values
            y = df[target_col].values
        else:
            st.warning("Please select exactly 2 features.")
            st.stop()
    else:
        st.warning("Upload a CSV to continue.")
        st.stop()
else:
    if dataset == "Circles":
        X, y = make_circles(n_samples=1000, noise=noise, random_state=42, factor=0.5)
    elif dataset == "Moons":
        X, y = make_moons(n_samples=1000, noise=noise, random_state=42)
    elif dataset == "Blobs":
        X, y = make_blobs(n_samples=1000, centers=2, cluster_std=noise+0.5, random_state=42)

# =======================
# Data Preparation
# =======================
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# =======================
# Regularizer
# =======================
if reg_choice == "L1" and reg_rate is not None:
    reg = l1(reg_rate)
elif reg_choice == "L2" and reg_rate is not None:
    reg = l2(reg_rate)
elif reg_choice == "ElasticNet" and reg_rate is not None:
    reg = l1_l2(l1=reg_rate, l2=reg_rate)
else:
    reg = None

# =======================
# Train Model
# =======================
if st.sidebar.button("🚀 Train Model"):
    model = Sequential()
    model.add(InputLayer(shape=(2,)))

    # Hidden layers (robust handling)
    if len(nn) > 0:
        for units in nn:
            model.add(Dense(units=units, activation=af, kernel_regularizer=reg))

    # Output layer
    model.add(Dense(units=1, activation="sigmoid", kernel_regularizer=reg))

    optimizer = SGD(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    callbacks = []
    if es == "Yes":
        callbacks.append(EarlyStopping(monitor="val_loss", min_delta=min_delta, patience=patience, restore_best_weights=True))

    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),
                        batch_size=32, verbose=0, callbacks=callbacks)

    # =======================
    # Visualizations
    # =======================
    st.subheader("🌍 Decision Boundary")
    fig, ax = plt.subplots()
    plot_decision_regions(X, y, clf=model, legend=2)
    st.pyplot(fig)

    st.subheader("📉 Training vs Validation Loss")
    fig2, ax2 = plt.subplots()
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("📈 Training vs Validation Accuracy")
    fig3, ax3 = plt.subplots()
    ax3.plot(history.history["accuracy"], label="Train Accuracy")
    ax3.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax3.legend()
    st.pyplot(fig3)

    # =======================
    # Neural Network Architecture
    # =======================
    st.subheader("🧩 Neural Network Architecture")

    def visualize_nn(layers):
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")

        dot.node("Input", "Input Layer\nfeatures=2", shape="box", style="filled", color="lightblue")
        prev = "Input"
        for i, units in enumerate(layers):
            hid = f"H{i}"
            dot.node(hid, f"Hidden {i+1}\nunits={units}", shape="box", style="filled", color="lightgreen")
            dot.edge(prev, hid)
            prev = hid

        dot.node("Output", "Output Layer\nunits=1", shape="box", style="filled", color="lightcoral")
        dot.edge(prev, "Output")
        return dot

    st.graphviz_chart(visualize_nn(nn if nn else []))
