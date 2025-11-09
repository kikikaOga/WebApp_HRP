# app.py
#python - m streamlit run app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Health Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Feature List (23 Features) ------------------
# ⚠️ แก้ไข: ลดเหลือ 23 คอลัมน์ ตามที่โมเดล 'health_risk_model.pkl' คาดหวัง
# โดยการลบ 'diet_healthy' ออกจากการเข้ารหัส One-Hot Encoding ของตัวแปร 'diet'
MODEL_FEATURES = [
    'age', 'bmi', 'sleep', 'stress', 'smoking', 'alcohol', 'married',
    'gender_female', 'gender_male',
    'sugar_intake_high', 'sugar_intake_low', 'sugar_intake_medium',
    'diet_normal', 'diet_unhealthy', # 'diet_healthy' ถูกลบออกเพื่อให้เหลือ 23 ฟีเจอร์
    'profession_engineer', 'profession_farmer', 'profession_office_worker',
    'profession_student', 'profession_teacher',
    'exercise_high', 'exercise_low', 'exercise_medium', 'exercise_none'
]

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    # ตรวจสอบว่าไฟล์ 'health_risk_model.pkl' ถูกบันทึกและพร้อมใช้งาน
    try:
        model = joblib.load('health_risk_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล 'health_risk_model.pkl' โปรดตรวจสอบการบันทึกโมเดล")
        return None

model = load_model()

# ------------------ Title ------------------
st.title("🏥 Health Risk Prediction System")
st.write("""
ระบบนี้ใช้ Machine Learning (Decision Tree Classifier) เพื่อทำนายความเสี่ยงต่อสุขภาพของคุณ
โปรดกรอกข้อมูลส่วนตัวด้านล่างอย่างถูกต้อง
""")

st.divider()



# ------------------ SIDEBAR & Input Collection ------------------
with st.sidebar:
    st.header("📋 กรอกข้อมูลส่วนตัว")

    # Input 1: อายุ และ BMI
    age = st.slider("🎂 อายุ (ปี)", 18, 80, 40, 1)
    bmi = st.slider("📏 BMI", 15.0, 40.0, 24.0, 0.1)
    # height ถูกละเว้นจากการทำนาย เพราะโมเดลใช้ BMI
    height = st.slider("📏 ความสูง (cm)", 140, 200, 170, 1) 

    # Input 2: สูบบุหรี่ / แอลกอฮอล์ / สถานภาพสมรส
    smoking = st.selectbox("🚬 สูบบุหรี่", ["No", "Yes"])
    alcohol = st.selectbox("🍺 ดื่มแอลกอฮอล์", ["No", "Yes"])
    married = st.selectbox("💍 สถานภาพสมรส", ["No", "Yes"])

    # Input 3: การนอนหลับ / ความเครียด
    sleep = st.slider("😴 ชั่วโมงการนอนหลับต่อวัน", 4.0, 12.0, 7.5, 0.1)
    stress = st.slider("😫 ระดับความเครียด (1-10)", 1, 10, 5, 1)

    # Input 4: เพศ / อาหาร / อาชีพ / การออกกำลังกาย (Categorical)
    gender = st.selectbox("🚻 เพศ", ["Male", "Female"])
    sugar_intake = st.selectbox("🍬 ระดับการบริโภคน้ำตาล", ["Low", "Medium", "High"])
    diet = st.selectbox("🥦 รูปแบบการบริโภคอาหาร", ["Healthy", "Normal", "Unhealthy"])
    profession = st.selectbox("🧑‍💼 อาชีพ", ["Engineer", "Farmer", "Office Worker", "Student", "Teacher"])
    exercise = st.selectbox("🏋️ ระดับการออกกำลังกาย", ["None", "Low", "Medium", "High"])

# ------------------ MAIN CONTENT ------------------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📝 ข้อมูลที่กรอก")
    info_df = pd.DataFrame({
        "ข้อมูล": ["อายุ","BMI","ความสูง","สูบบุหรี่","ดื่มแอลกอฮอล์","สมรส","อาชีพ","ออกกำลังกาย","ชั่วโมงการนอน","ระดับน้ำตาล"],
        "ค่า": [age,bmi,height,smoking,alcohol,married,profession,exercise,sleep,sugar_intake]
    })
    st.table(info_df)
    st.divider()

# ------------------ Prediction Logic ------------------

if st.button("🎯 ทำนายความเสี่ยง" , use_container_width=True, type="primary"):
    if model is None:
        st.error("ไม่สามารถทำนายผลได้เนื่องจากโมเดลไม่พร้อมใช้งาน")
    else:
        # 1. Map Binary Features (0/1)
        smoking_val = 1 if smoking == "Yes" else 0
        alcohol_val = 1 if alcohol == "Yes" else 0
        married_val = 1 if married == "Yes" else 0
        
        # 2. Map Categorical Features (One-Hot Encoding)
        
        # Gender
        gender_male = 1 if gender == "Male" else 0
        gender_female = 1 if gender == "Female" else 0

        # Sugar Intake
        sugar_intake_high = 1 if sugar_intake == "High" else 0
        sugar_intake_low = 1 if sugar_intake == "Low" else 0
        sugar_intake_medium = 1 if sugar_intake == "Medium" else 0

        # Diet
        # ⚠️ แก้ไข: ไม่ต้องสร้าง diet_healthy เพราะถูกลบออกจาก MODEL_FEATURES แล้ว
        diet_normal = 1 if diet == "Normal" else 0
        diet_unhealthy = 1 if diet == "Unhealthy" else 0

        # Profession
        profession_engineer = 1 if profession == "Engineer" else 0
        profession_farmer = 1 if profession == "Farmer" else 0
        profession_office_worker = 1 if profession == "Office Worker" else 0
        profession_student = 1 if profession == "Student" else 0
        profession_teacher = 1 if profession == "Teacher" else 0

        # Exercise
        exercise_high = 1 if exercise == "High" else 0
        exercise_low = 1 if exercise == "Low" else 0
        exercise_medium = 1 if exercise == "Medium" else 0
        exercise_none = 1 if exercise == "None" else 0

        # 3. Create DataFrame (Input Sample) - ต้องมี 23 คอลัมน์เท่านั้น
        input_data = {
            'age': [age],
            'bmi': [bmi],
            'sleep': [sleep],
            'stress': [stress],
            'smoking': [smoking_val],
            'alcohol': [alcohol_val],
            'married': [married_val],
            'gender_female': [gender_female],
            'gender_male': [gender_male],
            'sugar_intake_high': [sugar_intake_high],
            'sugar_intake_low': [sugar_intake_low],
            'sugar_intake_medium': [sugar_intake_medium],
            'diet_normal': [diet_normal],
            'diet_unhealthy': [diet_unhealthy],
            'profession_engineer': [profession_engineer],
            'profession_farmer': [profession_farmer],
            'profession_office_worker': [profession_office_worker],
            'profession_student': [profession_student],
            'profession_teacher': [profession_teacher],
            'exercise_high': [exercise_high],
            'exercise_low': [exercise_low],
            'exercise_medium': [exercise_medium],
            'exercise_none': [exercise_none]
        }
        
        features_df = pd.DataFrame(input_data)
        
        # 4. **ขั้นตอนการแก้ไขปัญหาหลัก:** จัดเรียงคอลัมน์ให้ตรงตาม 23 ฟีเจอร์ของโมเดล
        features_aligned = features_df.reindex(columns=MODEL_FEATURES, fill_value=0)
        
        # 5. แปลงเป็น NumPy Array และ Reshape เป็น (1, 23)
        input_array = features_aligned.values.reshape(1, -1) 

        try:
            # ทำนายผล
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0]

            # แสดงผลลัพธ์
            risk_level = "🔴 ความเสี่ยงสูง" if prediction == 1 else "🟢 ความเสี่ยงต่ำ"
            confidence = probability[prediction] * 100

            st.success("✅ ทำนายผลสำเร็จ!")
            st.metric(label="ระดับความเสี่ยง", value=risk_level, 
                        # delta=f"ความมั่นใจ: {confidence:.1f}%"
                      )
            # st.write(f"- โอกาส ความเสี่ยงต่ำ : {probability[0]*100:.1f}%")
            # st.write(f"- โอกาส ความเสี่ยงสูง : {probability[1]*100:.1f}%")
            
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการทำนายผล: {e}")
            st.write(f"โปรดตรวจสอบว่าโมเดลใช้ 23 features นี้: {MODEL_FEATURES}")

# ------------------ Footer ------------------
st.markdown("""
---
**⚠️ สำคัญ:** ระบบนี้เป็นเพียงเครื่องมือช่วยในการประเมินเบื้องต้นเท่านั้น 
อย่าใช้แทนที่การตรวจสอบกับแพทย์ที่มีผู้เชี่ยวชาญ
📧 ติดต่อ: contact@healthprediction.com
""")     