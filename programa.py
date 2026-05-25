"""
🌸 Iris Species Classification Dashboard
Data Mining — Final Project
Universidad de la Costa (CUC) 2026-1

Authors:
- Rafael Ricardo Romo Restrepo
- Leonardo Avendaño Narváez

Professor: José Escorcia Gutierrez, Ph.D.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="Iris Classification — CUC",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 {
        font-size: 0.85rem;
        margin: 0;
        opacity: 0.9;
        font-weight: 400;
    }
    .metric-card h1 {
        font-size: 2.2rem;
        margin: 0.3rem 0 0 0;
        font-weight: 700;
    }
    .metric-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    .metric-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    .metric-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .pred-setosa {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
    }
    .pred-versicolor {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #2d3436;
    }
    .pred-virginica {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        color: #2d3436;
    }
    .authors-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .workflow-step {
        background: #f0f2f6;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# LOAD & PREPARE DATA
# ═══════════════════════════════════════════

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    return df, iris

@st.cache_resource
def train_models(df):
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="accuracy")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion": confusion_matrix(y_test, y_pred, labels=["setosa", "versicolor", "virginica"]),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    # PCA for 3D visualization
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(scaler.transform(X))

    return results, scaler, pca, X_train, X_test, y_train, y_test

df, iris = load_data()
results, scaler, pca, X_train, X_test, y_train, y_test = train_models(df)


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/220px-Iris_versicolor_3.jpg", width=200)
    st.markdown("## 🌸 Iris Classifier")
    st.markdown("""
    <div class='authors-box'>
        <strong>Authors:</strong><br>
        Rafael R. Romo Restrepo<br>
        Leonardo Avendaño Narváez<br><br>
        <strong>Course:</strong> Data Mining<br>
        <strong>Professor:</strong> José Escorcia G., Ph.D.<br>
        <strong>University:</strong> CUC — 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    selected_model = st.selectbox(
        "🤖 Select Classification Model",
        list(results.keys()),
        index=0
    )

    st.markdown("---")
    page = st.radio(
        "📄 Navigation",
        ["🔬 Workflow & Methodology",
         "📊 Model Metrics",
         "🎯 Predict Species",
         "📈 Data Exploration"],
        index=2
    )

model_data = results[selected_model]


# ═══════════════════════════════════════════
# PAGE: WORKFLOW
# ═══════════════════════════════════════════

if page == "🔬 Workflow & Methodology":
    st.markdown("# 🔬 Data Mining Workflow")
    st.markdown("Our end-to-end pipeline for the Iris species classification problem:")

    steps = [
        ("1️⃣ Data Understanding",
         "Load the Iris dataset (150 samples, 4 features, 3 classes). Explore the structure, distributions, and relationships between features using descriptive statistics and visualizations."),
        ("2️⃣ Data Preprocessing",
         "Check for missing values and outliers (the Iris dataset is clean). Apply StandardScaler normalization to ensure all features have zero mean and unit variance, which is critical for distance-based algorithms like SVM and KNN."),
        ("3️⃣ Data Splitting",
         "Split the dataset into 70% training and 30% testing sets using stratified sampling to preserve the class distribution in both subsets."),
        ("4️⃣ Model Selection & Training",
         "Train 4 classification algorithms: Random Forest (ensemble of decision trees), SVM with RBF kernel (finds optimal hyperplane), K-Nearest Neighbors (instance-based learning), and Gradient Boosting (sequential ensemble). Each model is selected to represent a different learning paradigm."),
        ("5️⃣ Model Evaluation",
         "Evaluate each model using Accuracy, Precision, Recall, and F1-Score on the test set. Apply 5-fold Stratified Cross-Validation on the training set to assess generalization and detect overfitting."),
        ("6️⃣ Prediction & Visualization",
         "Deploy an interactive prediction panel where users input flower measurements and receive species predictions. Use PCA to project features into 3D space for visualization of the prediction relative to the dataset."),
    ]

    for title, desc in steps:
        st.markdown(f"""
        <div class='workflow-step'>
            <strong>{title}</strong><br>
            {desc}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Why Random Forest as Default?")
    st.markdown("""
    Random Forest was selected as the default model because:
    - It handles non-linear decision boundaries well
    - It's robust to overfitting due to ensemble averaging
    - It provides feature importance rankings
    - It requires minimal hyperparameter tuning
    - It performs consistently well on small-to-medium datasets like Iris

    However, the dashboard allows comparing 4 different algorithms to demonstrate the impact of model selection on classification performance.
    """)

    # Feature importance (only for RF and GB)
    if selected_model in ["Random Forest", "Gradient Boosting"]:
        st.markdown(f"### 🌿 Feature Importance ({selected_model})")
        importances = model_data["model"].feature_importances_
        feat_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        fig_imp = px.bar(
            x=importances, y=feat_names,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=importances,
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(
            height=300, showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════════════════════════════
# PAGE: MODEL METRICS
# ═══════════════════════════════════════════

elif page == "📊 Model Metrics":
    st.markdown(f"# 📊 Model Metrics — {selected_model}")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Accuracy</h3>
            <h1>{model_data['accuracy']:.1%}</h1>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card metric-green'>
            <h3>Precision</h3>
            <h1>{model_data['precision']:.1%}</h1>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card metric-blue'>
            <h3>Recall</h3>
            <h1>{model_data['recall']:.1%}</h1>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card metric-orange'>
            <h3>F1-Score</h3>
            <h1>{model_data['f1']:.1%}</h1>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Cross-validation
    st.markdown(f"### 🔄 5-Fold Cross-Validation")
    st.info(f"*CV Accuracy:* {model_data['cv_mean']:.1%} ± {model_data['cv_std']:.1%}")

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown("### 🎯 Confusion Matrix")
        cm = model_data["confusion"]
        labels = ["Setosa", "Versicolor", "Virginica"]
        fig_cm = px.imshow(
            cm, x=labels, y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            aspect="equal"
        )
        fig_cm.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class metrics
    with col2:
        st.markdown("### 📋 Per-Class Report")
        report = model_data["report"]
        report_df = pd.DataFrame({
            "Species": ["Setosa", "Versicolor", "Virginica"],
            "Precision": [report["setosa"]["precision"], report["versicolor"]["precision"], report["virginica"]["precision"]],
            "Recall": [report["setosa"]["recall"], report["versicolor"]["recall"], report["virginica"]["recall"]],
            "F1-Score": [report["setosa"]["f1-score"], report["versicolor"]["f1-score"], report["virginica"]["f1-score"]],
            "Support": [int(report["setosa"]["support"]), int(report["versicolor"]["support"]), int(report["virginica"]["support"])],
        })
        st.dataframe(report_df.style.format({
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"
        }), use_container_width=True, hide_index=True)

    # Model comparison
    st.markdown("---")
    st.markdown("### 🏆 Model Comparison")
    comp_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [r["accuracy"] for r in results.values()],
        "Precision": [r["precision"] for r in results.values()],
        "Recall": [r["recall"] for r in results.values()],
        "F1-Score": [r["f1"] for r in results.values()],
        "CV Mean": [r["cv_mean"] for r in results.values()],
    })

    fig_comp = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric",
        barmode="group",
        color_discrete_sequence=["#667eea", "#11998e", "#4facfe", "#f093fb", "#ffa726"],
        labels={"Score": "Score", "Model": ""}
    )
    fig_comp.update_layout(
        height=400, yaxis=dict(range=[0.85, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════
# PAGE: PREDICT SPECIES
# ═══════════════════════════════════════════

elif page == "🎯 Predict Species":
    st.markdown("# 🎯 Predict Iris Species")
    st.markdown("Enter the flower measurements to get a predicted species and see it in 3D space.")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.markdown("### 📏 Input Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

        # Predict
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        prediction = model_data["model"].predict(input_scaled)[0]

        # Probabilities
        if hasattr(model_data["model"], "predict_proba"):
            proba = model_data["model"].predict_proba(input_scaled)[0]
            proba_dict = dict(zip(["setosa", "versicolor", "virginica"], proba))
        else:
            proba_dict = None

        # Prediction display
        color_map = {"setosa": "pred-setosa", "versicolor": "pred-versicolor", "virginica": "pred-virginica"}
        emoji_map = {"setosa": "🌷", "versicolor": "🌺", "virginica": "🌻"}

        st.markdown(f"""
        <div class='prediction-box {color_map[prediction]}'>
            {emoji_map[prediction]} Predicted: <strong>Iris {prediction.capitalize()}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        if proba_dict:
            st.markdown("*Prediction Confidence:*")
            for species, prob in proba_dict.items():
                st.progress(float(prob), text=f"{species.capitalize()}: {prob:.1%}")

    with col_result:
        st.markdown("### 🌐 3D Scatter Plot — PCA Projection")

        # PCA transform dataset
        X_all = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        X_all_sc = scaler.transform(X_all)
        X_pca_all = pca.transform(X_all_sc)

        # PCA transform input
        input_pca = pca.transform(input_scaled)

        # Dataset scatter
        pca_df = pd.DataFrame(X_pca_all, columns=["PC1", "PC2", "PC3"])
        pca_df["Species"] = df["species"].values

        color_discrete = {"setosa": "#2ecc71", "versicolor": "#e67e22", "virginica": "#9b59b6"}

        fig_3d = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3",
            color="Species",
            color_discrete_map=color_discrete,
            opacity=0.6,
            symbol="Species",
            labels={"PC1": "Component 1", "PC2": "Component 2", "PC3": "Component 3"}
        )

        # Add predicted point
        fig_3d.add_trace(go.Scatter3d(
            x=[input_pca[0, 0]], y=[input_pca[0, 1]], z=[input_pca[0, 2]],
            mode="markers",
            marker=dict(size=14, color="red", symbol="diamond",
                        line=dict(width=3, color="white")),
            name=f"🔴 Prediction: {prediction.capitalize()}",
            showlegend=True
        ))

        fig_3d.update_layout(
            height=550,
            scene=dict(
                xaxis=dict(backgroundcolor="#f8f9fa"),
                yaxis=dict(backgroundcolor="#f8f9fa"),
                zaxis=dict(backgroundcolor="#f8f9fa"),
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # Input summary table
    st.markdown("---")
    st.markdown("### 📋 Input Summary")
    summary_df = pd.DataFrame({
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Your Input (cm)": [sepal_length, sepal_width, petal_length, petal_width],
        "Dataset Mean": [df["sepal_length"].mean(), df["sepal_width"].mean(),
                         df["petal_length"].mean(), df["petal_width"].mean()],
        "Dataset Std": [df["sepal_length"].std(), df["sepal_width"].std(),
                        df["petal_length"].std(), df["petal_width"].std()],
    })
    st.dataframe(summary_df.style.format({
        "Your Input (cm)": "{:.1f}", "Dataset Mean": "{:.2f}", "Dataset Std": "{:.2f}"
    }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# PAGE: DATA EXPLORATION
# ═══════════════════════════════════════════

elif page == "📈 Data Exploration":
    st.markdown("# 📈 Data Exploration")

    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", 150)
    c2.metric("Features", 4)
    c3.metric("Classes", 3)

    # Show data
    with st.expander("🗂️ View Raw Dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Descriptive statistics
    with st.expander("📊 Descriptive Statistics"):
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Histograms
    with col1:
        st.markdown("### 📊 Feature Distributions")
        feature = st.selectbox("Select Feature",
                               ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                               format_func=lambda x: x.replace("_", " ").title())
        fig_hist = px.histogram(
            df, x=feature, color="species",
            barmode="overlay", opacity=0.7, nbins=25,
            color_discrete_map={"setosa": "#2ecc71", "versicolor": "#e67e22", "virginica": "#9b59b6"},
            labels={feature: feature.replace("_", " ").title(), "species": "Species"}
        )
        fig_hist.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Box plots
    with col2:
        st.markdown("### 📦 Box Plots")
        feature_box = st.selectbox("Select Feature ",
                                   ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                                   format_func=lambda x: x.replace("_", " ").title(),
                                   key="box")
        fig_box = px.box(
            df, x="species", y=feature_box, color="species",
            color_discrete_map={"setosa": "#2ecc71", "versicolor": "#e67e22", "virginica": "#9b59b6"},
            labels={feature_box: feature_box.replace("_", " ").title(), "species": "Species"}
        )
        fig_box.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Scatter Matrix
    st.markdown("### 🔗 Scatter Matrix (Pair Plot)")
    fig_scatter = px.scatter_matrix(
        df,
        dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        color="species",
        color_discrete_map={"setosa": "#2ecc71", "versicolor": "#e67e22", "virginica": "#9b59b6"},
        labels={col: col.replace("_", " ").title() for col in df.columns[:4]},
        opacity=0.7
    )
    fig_scatter.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))
    fig_scatter.update_traces(diagonal_visible=True, showupperhalf=True)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation heatmap
    st.markdown("### 🌡️ Correlation Heatmap")
    corr = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].corr()
    fig_corr = px.imshow(
        corr,
        x=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        y=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="equal",
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Violin plots
    st.markdown("### 🎻 Violin Plots — All Features")
    df_melt = df.melt(id_vars="species", var_name="Feature", value_name="Value")
    df_melt["Feature"] = df_melt["Feature"].str.replace("_", " ").str.title()
    fig_violin = px.violin(
        df_melt, x="Feature", y="Value", color="species",
        box=True, points="outliers",
        color_discrete_map={"setosa": "#2ecc71", "versicolor": "#e67e22", "virginica": "#9b59b6"},
    )
    fig_violin.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_violin, use_container_width=True)


# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85rem;'>
    🌸 Iris Species Classification — Data Mining Final Project<br>
    Rafael R. Romo Restrepo & Leonardo Avendaño Narváez<br>
    Universidad de la Costa (CUC) — 2026
</div>
""", unsafe_allow_html=True)