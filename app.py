# ==========================================================
# PASSAGE Hospital Stable Edition
# Safe Version (No Secrets Crash)
# ==========================================================

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import tempfile
import bcrypt
# ===== SAFE ML IMPORT =====
ML_AVAILABLE = True
try:
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ioff()
    from sklearn.linear_model import LogisticRegression
    try:
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ioff()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss
    from sklearn.model_selection import train_test_split
except Exception:
    ML_AVAILABLE = False
    from sklearn.model_selection import train_test_split
except Exception:
    ML_AVAILABLE = False
# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(page_title="PASSAGE Hospital Edition", layout="wide")
# ===== PROFESSIONAL HEADER =====
st.markdown("""
<style>
.main-title {font-size:28px;font-weight:700;}
.subtitle {color:gray;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='main-title'>PASSAGE Clinical Decision Support System</div>
<div class='subtitle'>
Cholangiocarcinoma Risk Stratification Platform | Hospital Edition
</div>
<hr>
""", unsafe_allow_html=True)

st.info(
"Clinical Decision Support Tool — Final diagnosis must be made by qualified physicians."
)
# Safe DB fallback (ไม่พังถ้าไม่มี secrets)
DATABASE_URL = st.secrets.get("DATABASE_URL", "sqlite:///passage_local.db")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()
# ==========================================================
# MODELS
# ==========================================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password = Column(String)
    role = Column(String)

class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True)
    patient_name = Column(String)
    age = Column(Integer)
    red_flags = Column(Integer)
    ca19_9 = Column(Float)
    risk_level = Column(String)
    followup_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    user_email = Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ==========================================================
# INITIAL ADMIN (สร้างครั้งแรก)
# ==========================================================

def create_default_admin():
    if session.query(User).count() == 0:
        hashed = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        admin = User(email="admin@passage.local", password=hashed, role="admin")
        session.add(admin)
        session.commit()
        st.caption(
            f"Assessment recorded at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

create_default_admin()

# ==========================================================
# LOGIN
# ==========================================================

def authenticate(email, password):
    user = session.query(User).filter_by(email=email).first()
    if user and bcrypt.checkpw(password.encode(), user.password.encode()):
        return user
    return None

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("PASSAGE Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate(email, password)
        if user:
            st.session_state.user = user
            session.add(AuditLog(user_email=email, action="Login"))
            session.commit()
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

user = st.session_state.user
st.sidebar.success(f"{user.email} ({user.role})")
st.sidebar.markdown("---")
st.sidebar.success("System Status: Operational")

if "postgresql" in DATABASE_URL:
    st.sidebar.caption("Database: PostgreSQL (Production)")
else:
    st.sidebar.caption("Database: Local Development Mode")

# ==========================================================
# RISK ENGINE
# ==========================================================

def calculate_risk_protocol(
    age,
    raw_fish,
    psc,
    abnormal_lft,
    red_flags,
    ca19_9,
    alp,
    bilirubin,
    us_dilation,
    us_mass
):

    score = 0

    # Epidemiologic risk
    if age >= 40:
        score += 1
    if raw_fish:
        score += 2
    if psc:
        score += 3

    # Clinical signs
    score += red_flags * 2

    # Lab abnormalities
    if abnormal_lft:
        score += 2
    if ca19_9 > 37:
        score += 2
    if ca19_9 > 100:
        score += 3
    if alp > 147:
        score += 1
    if bilirubin > 1.2:
        score += 1

    # Imaging
    if us_dilation:
        score += 3
    if us_mass:
        score += 5

    # Risk stratification
    if score >= 12:
        return "High Suspicion"
    elif score >= 6:
        return "Intermediate Risk"
    else:
        return "Low Risk"
# ==========================================================
# AUTO ML ENGINE (PASSAGE-CDS)
# ==========================================================

@st.cache_data(show_spinner=False)
def train_ml_model(record_count):

    records = session.query(Assessment).all()

    if len(records) < 5:
        return None

    df = pd.DataFrame([{
        "age": r.age,
        "red_flags": r.red_flags,
        "ca19_9": r.ca19_9,
        "target": 1 if r.risk_level == "High Suspicion" else 0
    } for r in records])

    X = df[["age", "red_flags", "ca19_9"]]
    y = df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model
# ==========================================================
# MENU
# ==========================================================

menu = st.sidebar.selectbox(
    "Navigation",
    ["New Screening", "Recall List", "Dashboard", "Clinical Protocol Guide", "AI Analytics"]
)

# ==========================================================
# NEW SCREENING – PROTOCOL VERSION
# ==========================================================

if menu == "New Screening":

    st.header("CCA Screening (Protocol-Based)")

    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input("Patient Name")
        age = st.number_input("Age", 20, 100)

        st.subheader("Epidemiologic Risk")
        raw_fish = st.checkbox("History of Raw Fish Consumption")
        psc = st.checkbox("Primary Sclerosing Cholangitis (PSC)")
        abnormal_lft = st.checkbox("Abnormal Liver Function Test")
        red_flags = st.slider(
            "Red Flag Symptoms (0–5)",
            0, 5,
            help="Jaundice, Weight loss, RUQ pain, Anorexia, Cholangitis"
        )

    with col2:
        st.subheader("Tumor Markers")
        ca19_9 = st.number_input("CA19-9 (U/mL)", 0.0)
        alp = st.number_input("ALP (U/L)", 0.0)
        bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.0)

        st.subheader("Ultrasound Findings")
        us_dilation = st.checkbox("Bile Duct Dilation")
        us_mass = st.checkbox("Liver Mass Detected")

    if st.button("Evaluate Risk"):

        risk = calculate_risk_protocol(
            age,
            raw_fish,
            psc,
            abnormal_lft,
            red_flags,
            ca19_9,
            alp,
            bilirubin,
            us_dilation,
            us_mass
        )

        # =====================================
        # REAL ML PROBABILITY (AUTO MODEL)
        # =====================================
        record_count = session.query(Assessment).count()
        model = None
        if record_count >= 5:
            model = train_ml_model(record_count)
        
        if model is not None:
        
            X_new = pd.DataFrame([{
                "age": age,
                "red_flags": red_flags,
                "ca19_9": ca19_9
            }])
        
            probability = model.predict_proba(X_new)[0][1]
        
            st.metric(
                "CCA Probability (ML Model)",
                f"{probability:.2%}"
            )
        
        else:
            st.info("Model will activate after ≥5 patients.")
        
        # Follow-up scheduling
        followup = None
        if risk == "Intermediate Risk":
            followup = datetime.utcnow() + timedelta(days=90)
        elif risk == "Low Risk":
            followup = datetime.utcnow() + timedelta(days=365)

        assessment = Assessment(
            patient_name=patient_name,
            age=age,
            red_flags=red_flags,
            ca19_9=ca19_9,
            risk_level=risk,
            followup_date=followup
        )

        session.add(assessment)
        session.add(AuditLog(
            user_email=user.email,
            action=f"Protocol screening for {patient_name}"
        ))
        session.commit()

        st.markdown("---")
        st.markdown("### Clinical Interpretation")

        interpret_map = {
            "High Suspicion":
                "Findings strongly suggest possible cholangiocarcinoma. Immediate specialist referral recommended.",
            "Intermediate Risk":
                "Abnormal risk profile detected. Imaging surveillance advised.",
            "Low Risk":
                "No significant risk detected at this time."
        }

        st.info(interpret_map[risk])

        if risk == "High Suspicion":
            st.error("High Suspicion of CCA")
            st.write("### Recommended Action:")
            st.write("• Urgent hepatobiliary referral")
            st.write("• Contrast-enhanced CT or MRI")
            st.write("• Multidisciplinary tumor board evaluation")

        elif risk == "Intermediate Risk":
            st.warning("Intermediate Risk")
            st.write("### Recommended Action:")
            st.write("• Ultrasound within 3 months")
            st.write("• Repeat CA19-9")
            st.write("• Monitor liver enzymes")

        else:
            st.success("Low Risk")
            st.write("### Recommended Action:")
            st.write("• Annual surveillance")
            st.write("• Lifestyle modification")

        # PDF Referral for High Suspicion
        if risk == "High Suspicion":

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            doc = SimpleDocTemplate(temp.name)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Cholangiocarcinoma Referral Letter", styles["Title"]))
            elements.append(Spacer(1, 0.3 * inch))
            elements.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
            elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
            elements.append(Paragraph("Risk Classification: HIGH SUSPICION", styles["Normal"]))
            elements.append(Paragraph("Recommendation: Urgent hepatobiliary evaluation.", styles["Normal"]))

            doc.build(elements)

            with open(temp.name, "rb") as f:
                st.download_button(
                    "Download Referral PDF",
                    f,
                    file_name="CCA_Referral.pdf",
                    mime="application/pdf"
                )

        st.caption("This tool supports clinical decision-making and does not replace physician judgment.")

# ==========================================================
# RECALL LIST
# ==========================================================

if menu == "Recall List":

    st.header("Recall Scheduler")

    today = datetime.utcnow()

    recalls = session.query(Assessment).filter(
        Assessment.followup_date != None,
        Assessment.followup_date <= today
    ).all()

    if not recalls:
        st.success("No patients due for recall")
    else:
        df = pd.DataFrame([{
            "Patient": r.patient_name,
            "Risk": r.risk_level,
            "Follow-up Date": r.followup_date
        } for r in recalls])
        st.dataframe(df)

# ==========================================================
# DASHBOARD
# ==========================================================

if menu == "Dashboard":

    st.header("Screening Dashboard")

    records = session.query(Assessment).all()

    if not records:
        st.info("No screening data available.")
    else:
        df = pd.DataFrame([{"risk": r.risk_level} for r in records])

        # ===== KPI =====
        high = (df["risk"] == "High Suspicion").sum()
        intermediate = (df["risk"] == "Intermediate Risk").sum()
        low = (df["risk"] == "Low Risk").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("High Suspicion Cases", high)
        c2.metric("Intermediate Risk", intermediate)
        c3.metric("Low Risk", low)

        st.markdown("---")

        col1, col2 = st.columns(2)
        col1.metric("Total Screenings", len(df))
        col2.metric(
            "High Suspicion %",
            round((df["risk"] == "High Suspicion").mean() * 100, 1)
        )

        st.bar_chart(df["risk"].value_counts())
        st.markdown("---")
        st.subheader("Research Dataset Export")
        
        export_df = pd.DataFrame([{
            "patient": r.patient_name,
            "age": r.age,
            "red_flags": r.red_flags,
            "ca19_9": r.ca19_9,
            "risk": r.risk_level,
            "created_at": r.created_at
        } for r in records])
        
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            "Download Research Dataset (CSV)",
            csv,
            file_name="PASSAGE_CCA_dataset.csv",
            mime="text/csv"
)


# ==========================================================
# CLINICAL PROTOCOL GUIDE
# ==========================================================

elif menu == "Clinical Protocol Guide":

    st.header("CCA Screening Protocol Reference")

    # ===============================
    # Tumor Marker
    # ===============================
    st.subheader("Tumor Marker Reference")

    marker_table = pd.DataFrame({
        "Marker": ["CA19-9", "ALP", "Total Bilirubin"],
        "Normal Range": ["< 37 U/mL", "44–147 U/L", "0.1–1.2 mg/dL"],
        "Clinical Significance": [
            "Elevated in CCA; >100 U/mL increases suspicion",
            "Elevated in biliary obstruction",
            "Elevated in obstructive jaundice"
        ]
    })

    st.dataframe(marker_table, use_container_width=True)

    # ===============================
    # Imaging
    # ===============================
    st.subheader("Imaging Red Flags")

    imaging_table = pd.DataFrame({
        "Finding": ["Bile Duct Dilation", "Liver Mass"],
        "Interpretation": [
            "Suggests obstructive pathology",
            "High suspicion for malignancy"
        ]
    })

    st.dataframe(imaging_table, use_container_width=True)

    # ===============================
    # Risk Summary
    # ===============================
    st.subheader("Risk Classification Summary")

    risk_table = pd.DataFrame({
        "Category": ["Low Risk", "Intermediate Risk", "High Suspicion"],
        "Recommended Action": [
            "Annual surveillance",
            "Ultrasound within 3 months",
            "Urgent CT/MRI + Specialist referral"
        ]
    })

    st.dataframe(risk_table, use_container_width=True)
# ==========================================================
# AI ANALYTICS – LOGISTIC REGRESSION MODEL
# ==========================================================

elif menu == "AI Analytics":

    st.header("AI Model Analytics (PASSAGE-CDS)")

    if not ML_AVAILABLE:
        st.error("Machine Learning modules not installed.")
        st.info("Check requirements.txt and rebuild environment.")
        st.stop()

    st.header("AI Model Analytics (PASSAGE-CDS)")

    records = session.query(Assessment).all()

    if len(records) < 5:
        st.warning("Need at least 5 records for model training.")
    else:

        # ==========================
        # Dataset preparation
        # ==========================
        df = pd.DataFrame([{
            "age": r.age,
            "red_flags": r.red_flags,
            "ca19_9": r.ca19_9,
            "target": 1 if r.risk_level=="High Suspicion" else 0
        } for r in records])

        X = df[["age","red_flags","ca19_9"]]
        y = df["target"]

        # ==========================
        # Train/Test split
        # ==========================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # ==========================
        # Probability prediction
        # ==========================
        y_prob = model.predict_proba(X_test)[:,1]

        # ==========================
        # ROC + AUC
        # ==========================
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        st.pyplot(fig)

        st.metric("Model AUC", round(roc_auc,3))

        # ==========================
        # Model coefficients
        # ==========================
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        })

        st.subheader("Model Explainability (Logistic Coefficients)")
        st.dataframe(coef_df, use_container_width=True)

        # ==========================
        # Population risk distribution
        # ==========================
        st.subheader("Population Risk Probability Distribution")

        probs_all = model.predict_proba(X)[:,1]

        fig2 = plt.figure()
        plt.hist(probs_all, bins=10)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Patients")

        st.pyplot(fig2)

        st.success("Model training completed successfully.")
        # ==========================================================
        # CLINICAL AI VALIDATION LAYER
        # ==========================================================
        
        st.markdown("---")
        st.header("Clinical Model Validation")
        # ===============================
        # Sensitivity / Specificity
        # ===============================
        st.subheader("Diagnostic Performance")
        
        y_pred = (y_prob >= 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
        
        c1, c2 = st.columns(2)
        c1.metric("Sensitivity (Recall)", f"{sensitivity:.2f}")
        c2.metric("Specificity", f"{specificity:.2f}")
        
        # ===============================
        # Calibration Curve
        # ===============================
        st.subheader("Calibration Curve (Model Reliability)")
        
        bins = pd.qcut(y_prob, q=5, duplicates="drop")
        
        calibration_df = pd.DataFrame({
            "prob": y_prob,
            "actual": y_test,
            "bin": bins
        })
        
        calibration = calibration_df.groupby("bin").mean()
        
        fig_cal = plt.figure()
        plt.plot(calibration["prob"], calibration["actual"], marker="o")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Observed Frequency")
        plt.title("Calibration Curve")
        
        st.pyplot(fig_cal)
        
        brier = brier_score_loss(y_test, y_prob)
        st.metric("Brier Score (Lower is better)", round(brier,3))
        
        # ===============================
        # Clinical Explainability
        # ===============================
        st.subheader("Clinical Explainability (Feature Impact)")
        
        coef_df["Odds Ratio"] = np.exp(coef_df["Coefficient"])
        
        st.dataframe(coef_df[["Feature","Coefficient","Odds Ratio"]],
                     use_container_width=True)
        
        st.info("""
        Interpretation Guide:
        
        • Odds Ratio > 1 → increases CCA probability  
        • Odds Ratio < 1 → protective association  
        • Larger magnitude = stronger clinical influence
        
        This explanation is derived from Logistic Regression coefficients
        and is interpretable for clinical decision-making.
        """)
# ==========================================================
# AUDIT LOG (Admin only)
# ==========================================================

if user.role == "admin":

    st.markdown("---")
    st.subheader("Audit Log")

    logs = session.query(AuditLog).all()

    df_logs = pd.DataFrame([{
        "User": l.user_email,
        "Action": l.action,
        "Time": l.timestamp
    } for l in logs])

    st.dataframe(df_logs)

st.markdown("---")
st.caption("PASSAGE Hospital Stable Edition")
