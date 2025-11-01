# app.py
import json
import io
import pandas as pd
import streamlit as st
from typing import List, Dict

# import your optimizer function (assumes optimizer.py in same folder)
from optimizer import solve_itinerary

st.set_page_config(page_title="Trip Optimizer (Streamlit)", layout="wide")

st.title("🧭 Trip Itinerary Optimizer (Streamlit)")
st.markdown(
    """
    อัปโหลดหรือกรอกข้อมูลโรงแรมและสถานที่ท่องเที่ยว แล้วกด **Run Optimization**  
    โปรดตรวจสอบว่าคุณได้ตั้งค่า `GEOAPIFY_API_KEY` ใน `st.secrets` (หรือ environment สำหรับการ deploy)
    """
)

# --- Helpers ---
def read_places_csv(uploaded_file, expected_cols):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")
        return None
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"CSV missing columns: {missing}. คอลัมน์ที่ต้องมี: {expected_cols}")
        return None
    return df

def df_to_place_list_hotels(df: pd.DataFrame) -> List[Dict]:
    return [
        {"name": str(r["name"]), "lat": float(r["lat"]), "lon": float(r["lon"])}
        for _, r in df.iterrows()
    ]

def df_to_place_list_attractions(df: pd.DataFrame) -> List[Dict]:
    # duration optional; default 1 hour
    return [
        {
            "name": str(r["name"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "duration": float(r.get("duration", 1) if "duration" in r.index else r.get("duration", 1))
        }
        for _, r in df.iterrows()
    ]

# --- Left column: data input ---
col1, col2 = st.columns([1.6, 1])

with col1:
    st.header("1) Input Data")

    st.subheader("Hotels (เลือกอย่างน้อย 1)")
    hotels_method = st.radio("Upload or Manual?", ("Upload CSV", "Manual entry"), key="hotels_method")
    hotels_list = []
    if hotels_method == "Upload CSV":
        uploaded_hotels = st.file_uploader("Upload hotels CSV (columns: name, lat, lon)", type=["csv"], key="hotels_csv")
        if uploaded_hotels:
            df_hotels = read_places_csv(uploaded_hotels, ["name", "lat", "lon"])
            if df_hotels is not None:
                st.dataframe(df_hotels)
                hotels_list = df_to_place_list_hotels(df_hotels)
    else:
        st.write("กรอกโรงแรม (กรอกอย่างน้อย 1 แถว)")
        hotels_manual = st.text_area("กรอกแต่ละแถว: name,lat,lon (ตัวอย่าง: MyHotel,13.7563,100.5018) — คั่นด้วยบรรทัดใหม่", height=120, key="hotels_manual")
        if hotels_manual.strip():
            rows = [r.strip() for r in hotels_manual.splitlines() if r.strip()]
            parsed = []
            for r in rows:
                try:
                    name, lat, lon = [x.strip() for x in r.split(",")]
                    parsed.append({"name": name, "lat": float(lat), "lon": float(lon)})
                except Exception as e:
                    st.error(f"Can't parse line: {r} — {e}")
            hotels_list = parsed
            if hotels_list:
                st.write(pd.DataFrame(hotels_list))

    st.subheader("Attractions (สถานที่ท่องเที่ยว)")
    attr_method = st.radio("Upload or Manual?", ("Upload CSV", "Manual entry"), key="attr_method")
    attractions_list = []
    if attr_method == "Upload CSV":
        uploaded_attr = st.file_uploader("Upload attractions CSV (columns: name, lat, lon, duration)", type=["csv"], key="attr_csv")
        if uploaded_attr:
            df_attr = read_places_csv(uploaded_attr, ["name", "lat", "lon"])
            if df_attr is not None:
                # duration optional
                if "duration" not in df_attr.columns:
                    df_attr["duration"] = 1.0
                st.dataframe(df_attr)
                attractions_list = df_to_place_list_attractions(df_attr)
    else:
        st.write("กรอกสถานที่ (ตัวอย่าง: Wat Arun,13.7437,100.4889,1.5)")
        attr_manual = st.text_area("กรอกแต่ละแถว: name,lat,lon,duration (duration เป็นชั่วโมง, default=1.0)", height=160, key="attr_manual")
        if attr_manual.strip():
            rows = [r.strip() for r in attr_manual.splitlines() if r.strip()]
            parsed = []
            for r in rows:
                try:
                    parts = [x.strip() for x in r.split(",")]
                    if len(parts) == 3:
                        name, lat, lon = parts
                        dur = 1.0
                    else:
                        name, lat, lon, dur = parts[:4]
                    parsed.append({"name": name, "lat": float(lat), "lon": float(lon), "duration": float(dur)})
                except Exception as e:
                    st.error(f"Can't parse line: {r} — {e}")
            attractions_list = parsed
            if attractions_list:
                st.write(pd.DataFrame(attractions_list))

with col2:
    st.header("2) Trip & Optimization Parameters")
    trip_days = st.number_input("Trip duration (days)", min_value=1, max_value=30, value=2, step=1)
    max_daily_hours = st.number_input("Max daily hours for travel+visit (hours)", min_value=1.0, max_value=24.0, value=8.0, step=0.5)
    flexible = st.checkbox("Allow daily limit flexible (soft constraint)", value=True, help="ถ้าเปิด จะอนุญาตให้ขยายเวลาเที่ยวต่อวันเมื่อจำเป็น (มี slack)")

    st.markdown("**Objective weights** (sum doesn't have to be 1 — we normalize inside optimizer)")
    distance_weight = st.slider("Distance weight (alpha)", 0.0, 1.0, 0.7, 0.05)
    time_balance_weight = st.slider("Time-balance weight (beta)", 0.0, 1.0, 0.3, 0.05)

    st.markdown("---")
    st.caption("Geoapify API key จะถูกอ่านจาก `st.secrets['GEOAPIFY_API_KEY']`. ถ้ายังไม่มี ให้เซ็ตใน Secrets (local: .streamlit/secrets.toml, หรือใน Streamlit Cloud)")
    show_debug = st.checkbox("Show debug prints (console)", value=False)

# --- Run optimization ---
st.markdown("---")
run_col1, run_col2 = st.columns([1, 2])
with run_col1:
    run_button = st.button("▶️ Run Optimization")

with run_col2:
    st.write("ตัวอย่างไฟล์ CSV (columns):")
    st.code("hotels.csv:\nname,lat,lon\nMyHotel,13.7563,100.5018", language="csv")
    st.code("attractions.csv:\nname,lat,lon,duration\nWat Arun,13.7437,100.4889,1.5", language="csv")

if run_button:
    # basic validation
    if not hotels_list:
        st.error("กรุณาใส่ข้อมูลโรงแรมอย่างน้อย 1 แห่ง")
    elif not attractions_list:
        st.error("กรุณาใส่สถานที่ท่องเที่ยวอย่างน้อย 1 แห่ง")
    else:
        # compose objective_weights dict expected by optimizer
        objective_weights = {
            "distance_weight": float(distance_weight),
            "time_balance_weight": float(time_balance_weight)
        }

        with st.spinner("Running optimizer — calling solve_itinerary() ..."):
            try:
                itineraries = solve_itinerary(
                    potential_hotels=hotels_list,
                    potential_attractions=attractions_list,
                    trip_duration_days=int(trip_days),
                    max_daily_hours=float(max_daily_hours),
                    is_daily_limit_flexible=bool(flexible),
                    objective_weights=objective_weights
                )
            except Exception as e:
                st.exception(f"Error while running optimizer: {e}")
                itineraries = None

        if itineraries is None:
            st.error("ไม่มีผลลัพธ์ — ตรวจสอบ log และ st.secrets['GEOAPIFY_API_KEY']")
        elif len(itineraries) == 0:
            st.warning("ไม่มี itinerary ถูกสร้าง — อาจเป็นเพราะข้อมูลไม่ครบหรือไม่มีสถานที่ที่ถูกต้อง")
        else:
            st.success("Optimization finished ✅")
            # Show results
            for it in itineraries:
                st.subheader(f"{it.get('title', 'Itinerary')} — Total distance: {it.get('total_distance', 'N/A')} km")
                days = it.get("daily_routes", [])
                for day in days:
                    day_idx = day.get("day")
                    with st.expander(f"Day {day_idx} — distance: {day.get('distance')} km — total time: {day.get('total_time')} hr", expanded=False):
                        st.write("Route:")
                        st.write(day.get("route"))
                        st.write({
                            "travel_time": day.get("travel_time"),
                            "visit_time": day.get("visit_time"),
                            "total_time": day.get("total_time"),
                            "distance": day.get("distance")
                        })

            # JSON / download
            pretty = json.dumps(itineraries, indent=2, ensure_ascii=False)
            st.download_button("📥 Download results (JSON)", data=pretty, file_name="itineraries.json", mime="application/json")

# --- Footer / extras ---
st.markdown("---")
st.markdown(
    """
    **Notes / Tips**
    - ตั้งค่า `st.secrets['GEOAPIFY_API_KEY']` ก่อนรัน (local dev: .streamlit/secrets.toml):
      ```
      GEOAPIFY_API_KEY = "your_api_key_here"
      ```
    - เมื่อ deploy บน Streamlit Cloud ให้เพิ่ม Secrets ผ่าน Settings -> Secrets.
    - ฟอร์แมต CSV ต้องมีคอลัมน์ `name,lat,lon` สำหรับโรงแรม และ `name,lat,lon,duration` สำหรับสถานที่ (duration เป็นชั่วโมง)
    - ถ้าต้องการให้ผมเพิ่มแผนที่ (map) หรือ visual ของเส้นทาง บอกได้ — ผมจะเพิ่มให้
    """
)
