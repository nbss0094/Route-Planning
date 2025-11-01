# app.py
import streamlit as st
from optimizer import solve_itinerary

st.set_page_config(page_title="Tour Route Optimizer", page_icon="🗺️", layout="wide")

st.title("🗺️ Multi-Day Tourism Route Optimization")
st.write("กรอกข้อมูลเพื่อสร้างเส้นทางท่องเที่ยวที่เหมาะสมที่สุด โดยใช้แบบจำลองทางคณิตศาสตร์ (MIP)")

# -----------------------------
# 1. Input Section
# -----------------------------
st.header("🔹 ข้อมูลเบื้องต้น")

col1, col2 = st.columns(2)

with col1:
    trip_days = st.number_input("จำนวนวันเดินทาง (Days)", min_value=1, max_value=14, value=3)
    max_daily_hours = st.number_input("จำนวนชั่วโมงต่อวัน (Hours per day)", min_value=1, max_value=24, value=8)
    flexible_limit = st.checkbox("อนุญาตให้ขยายเวลาในแต่ละวันได้ (Flexible Daily Limit)", value=False)

with col2:
    st.subheader("⚖️ การถ่วงน้ำหนักวัตถุประสงค์")
    distance_weight = st.slider("Distance Weight (ระยะทาง)", 0.0, 1.0, 0.7)
    balance_weight = st.slider("Time Balance Weight (สมดุลเวลา)", 0.0, 1.0, 0.3)

# -----------------------------
# 2. Input Location Section
# -----------------------------
st.header("📍 ข้อมูลสถานที่")

st.markdown("**กรอกชื่อโรงแรม (อย่างน้อย 1 แห่ง)**")
hotel_name = st.text_input("ชื่อโรงแรม", placeholder="เช่น Le Méridien Chiang Mai")

st.markdown("**กรอกสถานที่ท่องเที่ยว (อย่างน้อย 1 แห่ง)**")
attractions_input = st.text_area("รายชื่อสถานที่ (คั่นด้วยบรรทัดใหม่)", placeholder="Doi Suthep\nNimman Road\nChiang Mai Night Bazaar")

# เวลาในการเที่ยวแต่ละสถานที่ (กำหนดเป็นชั่วโมง)
default_duration = st.number_input("เวลาที่ใช้ต่อสถานที่ (ชั่วโมง)", min_value=0.5, max_value=10.0, value=1.0)

# -----------------------------
# 3. Process Button
# -----------------------------
if st.button("🚀 Optimize Route"):
    if not hotel_name:
        st.error("⚠️ กรุณากรอกชื่อโรงแรมอย่างน้อย 1 แห่งก่อน")
    elif not attractions_input.strip():
        st.error("⚠️ กรุณากรอกชื่อสถานที่ท่องเที่ยวอย่างน้อย 1 แห่งก่อน")
    else:
        with st.spinner("🔍 กำลังคำนวณเส้นทางที่เหมาะสมที่สุด... โปรดรอสักครู่"):
            potential_hotels = [{"name": hotel_name}]
            attractions_list = [
                {"name": name.strip(), "duration": default_duration}
                for name in attractions_input.split("\n")
                if name.strip()
            ]

            objective_weights = {
                "distance_weight": distance_weight,
                "time_balance_weight": balance_weight
            }

            try:
                itineraries = solve_itinerary(
                    potential_hotels=potential_hotels,
                    potential_attractions=attractions_list,
                    trip_duration_days=trip_days,
                    max_daily_hours=max_daily_hours,
                    is_daily_limit_flexible=flexible_limit,
                    objective_weights=objective_weights
                )

                if not itineraries:
                    st.warning("ไม่พบเส้นทางที่เหมาะสม กรุณาลองใหม่อีกครั้ง")
                else:
                    st.success("🎉 ได้ผลลัพธ์แล้ว! ดูรายละเอียดด้านล่าง")

                    for itinerary in itineraries:
                        st.subheader(f"🧭 {itinerary['title']}")
                        st.write(f"**ระยะทางรวมทั้งหมด:** {itinerary['total_distance']} km")
                        st.write(f"**เวลารวมทั้งหมด:** {itinerary['total_time']} ชั่วโมง")

                        for day in itinerary["daily_routes"]:
                            st.markdown(f"### 📅 วันที่ {day['day']}")
                            st.write("เส้นทาง:")
                            st.write(" → ".join(day["route"]))
                            st.write(f"🕒 Travel: {day['travel_time']} hr | Visit: {day['visit_time']} hr | Total: {day['total_time']} hr")
                            st.write(f"📏 Distance: {day['distance']} km")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการรันโปรแกรม: {e}")
