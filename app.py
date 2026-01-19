import streamlit as st
from pipelines.prediction_pipeline import PredictionPipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>

    /* Title styling */
h1 {
    text-align: center;
    color: #ec4899 !important; /* force full pink */
    font-weight: 800;
    text-shadow: none !important; /* remove shadows */
    background: none !important; /* remove any background gradient */
}

    /* Button styling */
    div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #312e81, #1e3a8a, #2563eb);
    color: #f8fafc;
    font-size: 18px;
    font-weight: 600;
    border-radius: 12px;
    padding: 12px;
    border: none;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
    transition: all 0.3s ease-in-out;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1e3a8a, #2563eb, #3b82f6);
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.55);
    transform: translateY(-1px);
}

    /* Custom prediction result card */
    .result-card {
        margin-top: 20px;
        padding: 18px;
        border-radius: 12px;
        background: linear-gradient(90deg, #fdf2f8, #fce7f3);
        border-left: 6px solid #ec4899;
        font-size: 18px;
        font-weight: 600;
        color: #9d174d;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Overview")

st.sidebar.markdown(
    """
    **👤 Developed by:**  
    **Vashu Choudhary**

    **🧠 Project:**  
    Customer Segmentation using Machine Learning

    **⚙️ Technologies Used:**  
    - Python  
    - Pandas & NumPy  
    - Scikit-Learn 
    - Feature Engineering & Feature Selection 
    - KMeans Clustering  
    - Streamlit  
    - Gradient Boosting
    - Machine Learning Pipeline
    - Pickle 

    **🚀 Future Enhancements:**  
    - Cluster-wise business recommendations  
    - Visualization of customer clusters  
    - Distance-to-centroid confidence score  
    - Real-time customer profiling dashboard  

    ---
    💡 *Built for real-world ML deployment*
    """
)

# ---------------- ENCODING MAPS ----------------
EDUCATION_MAP = {
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4
}

MARITAL_STATUS_MAP = {
    "Married": 1,
    "Together": 1,
    "Absurd": 0,
    "Widow": 0,
    "YOLO": 0,
    "Divorced": 0,
    "Single": 0,
    "Alone": 0
}

CLUSTER_MAP = {
    0: "Medium Income & Medium Spending Customers",
    1: "Regular Customers with Low Spending",
    2: "High Value but Low Engagement Customers"
}

# ---------------- MAIN CARD ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.title("Customer Segmentation Predictor")

Age = st.number_input("Age", 18, 100, step=1)

Education_str = st.selectbox("Education", EDUCATION_MAP.keys())
Marital_Status_str = st.selectbox("Marital Status", MARITAL_STATUS_MAP.keys())

Parental_Status = st.selectbox("Parental Status", [0, 1])
Children = st.number_input("Children", 0, step=1)

Income = st.number_input("Income", 0.0)
Total_Spending = st.number_input("Total Spending", 0.0)

Days_as_Customer = st.number_input("Days as Customer", 0, step=1)
Recency = st.number_input("Recency", 0, step=1)

Wines = st.number_input("Wines", 0)
Fruits = st.number_input("Fruits", 0)
Meat = st.number_input("Meat", 0)
Fish = st.number_input("Fish", 0.0)
Sweets = st.number_input("Sweets", 0)
Gold = st.number_input("Gold", 0.0)

Web = st.number_input("Web Purchases", 0)
Catalog = st.number_input("Catalog Purchases", 0)
Store = st.number_input("Store Purchases", 0)

Discount_Purchases = st.number_input("Discount Purchases", 0)
Total_Promo = st.number_input("Total Promo", 0)
NumWebVisitsMonth = st.number_input("Web Visits / Month", 0)

# ---------------- PREDICTION ----------------
if st.button("Predict Customer Segment 🚀"):

    Education = EDUCATION_MAP[Education_str]
    Marital_Status = MARITAL_STATUS_MAP[Marital_Status_str]

    input_data = {
        "Age": Age,
        "Education": Education,
        "Marital Status": Marital_Status,
        "Parental Status": Parental_Status,
        "Children": Children,
        "Income": Income,
        "Total_Spending": Total_Spending,
        "Days_as_Customer": Days_as_Customer,
        "Recency": Recency,
        "Wines": Wines,
        "Fruits": Fruits,
        "Meat": Meat,
        "Fish": Fish,
        "Sweets": Sweets,
        "Gold": Gold,
        "Web": Web,
        "Catalog": Catalog,
        "Store": Store,
        "Discount Purchases": Discount_Purchases,
        "Total Promo": Total_Promo,
        "NumWebVisitsMonth": NumWebVisitsMonth
    }

    pipeline = PredictionPipeline()
    result = pipeline.predict(input_data)

    cluster_id = int(result)

    st.markdown(
        f"""
        <div class="result-card">
            🎯 Predicted Segment: <br>
            <strong>Cluster {cluster_id} — {CLUSTER_MAP[cluster_id]}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
