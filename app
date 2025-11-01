# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app (single-file) that includes the optimizer functions you provided
and a user interface to upload hotels/attractions, set parameters, call the
optimizer and download results.

IMPORTANT:
- Do NOT hardcode API keys. Set secrets either via .streamlit/secrets.toml
  for local development or via Streamlit Cloud's Secrets.
  Required secrets keys (optional depending on which API you use):
    GEOAPIFY_API_KEY
    GOOGLE_API_KEY

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

"""

import io
import json
import math
import random
from typing import List, Dict, Tuple

import pandas as pd
import requests
import streamlit as st

# ------------------------
# External solver import
# ------------------------
# We include the optimizer functions in this file so it's standalone.
# (Adapted from user's supplied optimizer.py content.)

# You will need `mip` installed for the optimizer to run.
from mip import Model, xsum, minimize, INTEGER, BINARY, CONTINUOUS

# Optional: googlemaps client can be used if you want geocoding/distance via Google
try:
    import googlemaps
except Exception:
    googlemaps = None

# ------------------------
# Helper: Google Maps client (optional)
# ------------------------

def get_google_client():
    key = None
    # Prefer Streamlit secrets
    if "GOOGLE_API_KEY" in st.secrets:
        key = st.secrets["GOOGLE_API_KEY"]
    else:
        # fallback to environment variable if you prefer (not set here)
        key = None
    if key and googlemaps:
        return googlemaps.Client(key=key)
    return None

# ------------------------
# Geocoding / road distance (optional)
# ------------------------

def geocode_address(address: str) -> dict | None:
    client = get_google_client()
    if client is None:
        st.warning("Google Maps client not configured (set secrets['GOOGLE_API_KEY']) — geocoding disabled")
        return None
    try:
        res = client.geocode(address)
        if res:
            return res[0]["geometry"]["location"]
    except Exception as e:
        st.error(f"Geocode error: {e}")
    return None


def road_distance(origin: Tuple[float, float], destination: Tuple[float, float]) -> Tuple[float, float] | Tuple[None, None]:
    client = get_google_client()
    if client is None:
        st.warning("Google Maps client not configured — road_distance disabled")
        return None, None
    try:
        result = client.distance_matrix(origins=[origin], destinations=[destination], mode="driving")
        element = result["rows"][0]["elements"][0]
        if element["status"] == "OK":
            distance_km = element["distance"]["value"] / 1000.0
            duration_hr = element["duration"]["value"] / 3600.0
            return distance_km, duration_hr
    except Exception as e:
        st.error(f"Google distance_matrix error: {e}")
    return None, None

# ------------------------
# Geoapify matrix (used in user's code) — recommended
# ------------------------

def get_travel_matrices(places: list[dict]) -> Tuple[list[list[float]], list[list[float]]]:
    """Call Geoapify Route Matrix API to get distance (km) and time (hours).
    Expects st.secrets['GEOAPIFY_API_KEY'] to exist.
    """
    if "GEOAPIFY_API_KEY" not in st.secrets:
        st.error("Missing GEOAPIFY_API_KEY in st.secrets — cannot call Geoapify")
        return [], []

    api_key = st.secrets["GEOAPIFY_API_KEY"]

    coords = [{"location": [float(place["lon"]), float(place["lat"]) ]} for place in places]
    api_url = f"https://api.geoapify.com/v1/routematrix?apiKey={api_key}"
    request_body = {"mode": "drive", "sources": coords, "targets": coords}

    try:
        resp = requests.post(api_url, json=request_body, timeout=30)
        resp.raise_for_status()
        resp_json = resp.json()

        distance_matrix = []
        time_matrix = []
        for row in resp_json.get("sources_to_targets", []):
            drow = []
            trow = []
            for cell in row:
                # cell.get values may be None—fallback to inf
                drow.append(cell.get("distance", float("inf")) / 1000.0)
                trow.append(cell.get("time", float("inf")) / 3600.0)
            distance_matrix.append(drow)
            time_matrix.append(trow)
        return distance_matrix, time_matrix
    except Exception as e:
        st.error(f"Geoapify request failed: {e}")
        return [], []

# ------------------------
# Main optimizer: run_optimize & solve_itinerary
# (kept largely as provided, with small fixes to avoid crashes if empty matrices)
# ------------------------

def run_optimize(data: dict) -> dict:
    # Unpack
    all_locations = data["all_locations"]
    hotel = data["hotel"]
    visiting_time = data["visiting_time"]
    d = data["d"]
    t = data["t"]
    day = data["day"]
    T_max = data["T_max"]
    flexible = data["flexible"]
    alpha = data["alpha"]
    beta = data["beta"]

    n = len(all_locations)
    N = set(range(n))
    # hotel could be a single name or list — we will treat hotel as index 0
    H = {0}
    A = N - H
    K = set(range(day))

    # If matrices are empty or wrong size, return trivial result
    if not d or len(d) != n or not t or len(t) != n:
        st.error("Distance/time matrices invalid — cannot run optimizer")
        return {
            "total_distance": 0,
            "objective_value": None,
            "daily_routes": [[] for _ in K],
            "daily_travel_time": [0 for _ in K],
            "daily_visit_time": [0 for _ in K],
            "daily_total_time_spent": [0 for _ in K],
            "daily_distance": [0 for _ in K]
        }

    model = Model()

    # Decision variables
    x = [[[model.add_var(var_type=BINARY) for k in K] for j in N] for i in N]
    y = [[model.add_var(var_type=BINARY) for k in K] for i in N]
    u = [[model.add_var(var_type=INTEGER, lb=0, ub=n-1) for k in K] for i in N]
    slack = [model.add_var(lb=0.0) for _ in K]
    Z = [model.add_var(var_type=CONTINUOUS) for _ in K]

    T = [model.add_var(lb=0) for _ in K]
    T_avg = model.add_var(lb=0)

    # Objectives
    objective_dist = xsum(d[i][j] * x[i][j][k] for k in K for i in N for j in N if i != j)
    objective_time_balance = xsum(Z[k] for k in K)
    objective_slack = xsum(slack[k] for k in K)
    objective_penalty = xsum(1 - xsum(y[i][k] for k in K) for i in A)

    # Bounds for normalization
    max_d = 0.0
    min_d = 0.0
    for i in range(n):
        row_values = [d[i][j] for j in range(n) if i != j]
        if row_values:
            max_row = max(row_values)
            min_row = min(row_values)
            max_d += max_row
            min_d += min_row

    objective_dist_min = min_d
    objective_dist_max = max_d if max_d > objective_dist_min else objective_dist_min + 1.0

    objective_time_balance_min = 0.0
    objective_time_balance_max = T_max * len(K)
    objective_slack_min = 0.0
    objective_slack_max = T_max * len(K)
    objective_penalty_min = 0.0
    objective_penalty_max = len(A)

    normalization_dist = (objective_dist - objective_dist_min) / (objective_dist_max - objective_dist_min)
    normalization_time_balance = (objective_time_balance - objective_time_balance_min) / max(1e-6, (objective_time_balance_max - objective_time_balance_min))
    normalization_slack = (objective_slack - objective_slack_min) / max(1e-6, (objective_slack_max - objective_slack_min))
    normalization_penalty = (objective_penalty - objective_penalty_min) / max(1e-6, (objective_penalty_max - objective_penalty_min))

    if flexible:
        model.objective = minimize((alpha * normalization_dist) + (beta * normalization_time_balance) + (normalization_slack))
    else:
        model.objective = minimize((alpha * normalization_dist) + (beta * normalization_time_balance) + (normalization_penalty))

    # Constraints (kept similar to provided code but guarded)
    for j in A:
        if flexible:
            model += xsum(x[i][j][k] for i in N if i != j for k in K) == 1
        else:
            model += xsum(x[i][j][k] for i in N if i != j for k in K) >= 0
            model += xsum(x[i][j][k] for i in N if i != j for k in K) <= 1

    for i in A:
        if flexible:
            model += xsum(x[i][j][k] for j in N if j != i for k in K) == 1
        else:
            model += xsum(x[i][j][k] for j in N if j != i for k in K) >= 0
            model += xsum(x[i][j][k] for j in N if j != i for k in K) <= 1

    for k in K:
        model += xsum(x[q][j][k] for q in H for j in A if j != q) == 1
        model += xsum(x[i][q][k] for q in H for i in A if i != q) == 1

    for k in K:
        for i in A:
            model += xsum(x[i][j][k] for j in N if j != i) == y[i][k]
            model += xsum(x[j][i][k] for j in N if j != i) == y[i][k]

    for k in K:
        for i in A:
            for j in A:
                if i != j:
                    model.add_constr(u[i][k] - u[j][k] + n * x[i][j][k] <= n - 1)

    for k in K:
        for q in H:
            model += u[q][k] == 0

    for i in A:
        for k in K:
            model.add_constr(u[i][k] >= y[i][k])
            model.add_constr(u[i][k] <= (n - 1) * (y[i][k]))

    for k in K:
        travel_term = xsum(t[i][j] * x[i][j][k] for i in N for j in N if i != j)
        visit_term = xsum(visiting_time[j] * xsum(x[i][j][k] for i in N if i != j) for j in N if j != 0)
        if flexible:
            model += travel_term + visit_term <= T_max + slack[list(K)[0]]  # simplified slack indexing
        else:
            model += travel_term + visit_term <= T_max

    for k in K:
        travel_time_k = xsum(t[i][j] * x[i][j][k] for i in N for j in N if i != j)
        visit_time_k = xsum(visiting_time[j] * y[j][k] for j in N)
        model += T[list(K)[0]] == travel_time_k + visit_time_k

    model += T_avg == xsum(T[k] for k in K) / max(1, len(K))

    for k in K:
        model += Z[k] >= T[list(K)[0]] - T_avg
        model += Z[k] >= T_avg - T[list(K)[0]]

    for k in range(1, len(K)):
        for q in H:
            model += xsum(x[q][j][k] for j in A) == xsum(x[i][q][k - 1] for i in A)

    for q in H:
        model += (xsum(x[q][j][0] for j in A)) - (xsum(x[i][q][0] for i in A)) == 0

    for k in K:
        for q in H:
            for r in H:
                if q != r:
                    model.add_constr(x[q][r][k] == 0)

    # Solve
    status = model.optimize()

    results = {
        "total_distance": 0,
        "objective_value": None,
        "daily_routes": [],
        "daily_travel_time": [],
        "daily_visit_time": [],
        "daily_total_time_spent": [],
        "daily_distance": [],
        "total_slack": None,
        "penalty_value": None,
    }

    if status == model.status.OPTIMAL or status == model.status.FEASIBLE:
        results["objective_value"] = model.objective_value
        total_dist = 0

        for k in K:
            day_dist = sum(d[i][j] * x[i][j][k].x for i in N for j in N if x[i][j][k].x is not None)
            total_dist += day_dist
            total_travel_time = sum(t[i][j] * x[i][j][k].x for i in N for j in N if i != j and x[i][j][k].x is not None)
            total_visit_time = sum(visiting_time[j] * sum(x[i][j][k].x for i in N if i != j and x[i][j][k].x is not None) for j in N if j not in H)
            total_time_spent = total_travel_time + total_visit_time

            # reconstruct route
            route = []
            start_hotel = None
            for q in H:
                for j in N:
                    if j != q and x[q][j][k].x is not None and x[q][j][k].x >= 0.5:
                        start_hotel = q
                        break
                if start_hotel is not None:
                    break

            if start_hotel is None:
                results["daily_routes"].append([])
                results["daily_travel_time"].append(0)
                results["daily_visit_time"].append(0)
                results["daily_total_time_spent"].append(0)
                results["daily_distance"].append(0)
                continue

            current = start_hotel
            while True:
                found = False
                for j in N:
                    if j != current and x[current][j][k].x is not None and x[current][j][k].x >= 0.5:
                        route.append((current, j))
                        if j in H and j == start_hotel:
                            found = False
                            break
                        current = j
                        found = True
                        break
                if not found:
                    break

            results["daily_routes"].append(route)
            results["daily_travel_time"].append(total_travel_time)
            results["daily_visit_time"].append(total_visit_time)
            results["daily_total_time_spent"].append(total_time_spent)
            results["daily_distance"].append(day_dist)

        results["total_distance"] = total_dist

    else:
        st.warning("No feasible solution found or solver failed")

    return results


def solve_itinerary(
    potential_hotels: list[dict],
    potential_attractions: list[dict],
    trip_duration_days: int,
    max_daily_hours: int,
    is_daily_limit_flexible: bool,
    objective_weights: dict,
) -> list[dict]:
    # Sanity checks
    if not potential_hotels:
        raise ValueError("Please provide at least one hotel")
    cleaned_attractions = [p for p in potential_attractions if p.get("name")]
    if not cleaned_attractions:
        return []

    selected_hotel = potential_hotels[0]
    all_places = [selected_hotel] + cleaned_attractions

    distance_matrix, time_matrix = get_travel_matrices(all_places)

    data = {
        "all_locations": [p["name"] for p in all_places],
        "hotel": selected_hotel["name"],
        "visiting_time": [p.get("duration", 1) for p in all_places],
        "d": distance_matrix,
        "t": time_matrix,
        "day": trip_duration_days,
        "T_max": max_daily_hours,
        "flexible": is_daily_limit_flexible,
        "alpha": objective_weights.get("distance_weight", 0.7),
        "beta": objective_weights.get("time_balance_weight", 0.3),
    }

    results = run_optimize(data)

    itineraries = []
    optimized_plan = []

    for k, route in enumerate(results.get("daily_routes", [])):
        day_route = []
        if route:
            start_point = all_places[route[0][0]]["name"]
            day_route.append(start_point)
            for (i, j) in route:
                day_route.append(all_places[j]["name"])
        else:
            day_route = ["No route available"]

        optimized_plan.append({
            "day": k + 1,
            "route": day_route,
            "travel_time": round(results["daily_travel_time"][k], 2) if len(results["daily_travel_time"]) > k else None,
            "visit_time": round(results["daily_visit_time"][k], 2) if len(results["daily_visit_time"]) > k else None,
            "total_time": round(results["daily_total_time_spent"][k], 2) if len(results["daily_total_time_spent"]) > k else None,
            "distance": round(results["daily_distance"][k], 2) if len(results["daily_distance"]) > k else None,
        })

    itineraries.append({
        "title": "Optimized Route",
        "total_distance": round(results.get("total_distance", 0), 2),
        "total_time": round(sum(results.get("daily_total_time_spent", [])), 2) if results.get("daily_total_time_spent") else None,
        "daily_routes": optimized_plan,
    })

    return itineraries

# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="Trip Optimizer", layout="wide")
st.title("🧭 Trip Itinerary Optimizer")

st.markdown("Upload hotels and attractions, set parameters, then run the optimizer.")

col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.header("Input data")
    hotels_method = st.radio("Hotels: Upload or Manual", ("Upload CSV", "Manual entry"), key="hotels_method")
    hotels_list = []
    if hotels_method == "Upload CSV":
        uploaded_hotels = st.file_uploader("Upload hotels CSV (name,lat,lon)", type=["csv"], key="hotels_csv")
        if uploaded_hotels:
            try:
                dfh = pd.read_csv(uploaded_hotels)
                if set(["name", "lat", "lon"]).issubset(dfh.columns):
                    hotels_list = [{"name": r["name"], "lat": float(r["lat"]), "lon": float(r["lon"])} for _, r in dfh.iterrows()]
                    st.dataframe(dfh)
                else:
                    st.error("Hotels CSV must contain columns: name, lat, lon")
            except Exception as e:
                st.error(f"Cannot read hotels CSV: {e}")
    else:
        manual_hotels = st.text_area("One per line: name,lat,lon", height=120, key="hotels_manual")
        if manual_hotels.strip():
            rows = [r.strip() for r in manual_hotels.splitlines() if r.strip()]
            parsed = []
            for row in rows:
                try:
                    name, lat, lon = [x.strip() for x in row.split(",")[:3]]
                    parsed.append({"name": name, "lat": float(lat), "lon": float(lon)})
                except Exception as e:
                    st.error(f"Can't parse hotel line: {row} — {e}")
            hotels_list = parsed
            if hotels_list:
                st.write(pd.DataFrame(hotels_list))

    st.subheader("Attractions")
    attr_method = st.radio("Attractions: Upload or Manual", ("Upload CSV", "Manual entry"), key="attr_method")
    attractions_list = []
    if attr_method == "Upload CSV":
        uploaded_attr = st.file_uploader("Upload attractions CSV (name,lat,lon,duration)", type=["csv"], key="attr_csv")
        if uploaded_attr:
            try:
                dfa = pd.read_csv(uploaded_attr)
                if set(["name", "lat", "lon"]).issubset(dfa.columns):
                    if "duration" not in dfa.columns:
                        dfa["duration"] = 1.0
                    attractions_list = [{"name": r["name"], "lat": float(r["lat"]), "lon": float(r["lon"]), "duration": float(r["duration"])} for _, r in dfa.iterrows()]
                    st.dataframe(dfa)
                else:
                    st.error("Attractions CSV must contain columns: name, lat, lon (duration optional)")
            except Exception as e:
                st.error(f"Cannot read attractions CSV: {e}")
    else:
        manual_attr = st.text_area("One per line: name,lat,lon,duration(optional)", height=160, key="attr_manual")
        if manual_attr.strip():
            rows = [r.strip() for r in manual_attr.splitlines() if r.strip()]
            parsed = []
            for row in rows:
                try:
                    parts = [x.strip() for x in row.split(",")]
                    if len(parts) >= 3:
                        name = parts[0]
                        lat = float(parts[1])
                        lon = float(parts[2])
                        dur = float(parts[3]) if len(parts) > 3 else 1.0
                        parsed.append({"name": name, "lat": lat, "lon": lon, "duration": dur})
                except Exception as e:
                    st.error(f"Can't parse attraction line: {row} — {e}")
            attractions_list = parsed
            if attractions_list:
                st.write(pd.DataFrame(attractions_list))

with col_right:
    st.header("Trip parameters")
    trip_days = st.number_input("Trip duration (days)", min_value=1, max_value=30, value=2, step=1)
    max_daily_hours = st.number_input("Max daily hours (travel + visit)", min_value=1.0, max_value=24.0, value=8.0, step=0.5)
    flexible = st.checkbox("Allow daily limit flexible (slack)", value=True)

    st.markdown("**Objective weights**")
    distance_weight = st.slider("Distance weight", 0.0, 1.0, 0.7, 0.05)
    time_balance_weight = st.slider("Time-balance weight", 0.0, 1.0, 0.3, 0.05)

    st.caption("Make sure GEOAPIFY_API_KEY is set in st.secrets for travel matrices")

# Run
st.markdown("---")
if st.button("Run Optimization"):
    if not hotels_list:
        st.error("Please provide at least 1 hotel")
    elif not attractions_list:
        st.error("Please provide at least 1 attraction")
    else:
        objective_weights = {"distance_weight": float(distance_weight), "time_balance_weight": float(time_balance_weight)}
        with st.spinner("Calling solve_itinerary..."):
            try:
                itineraries = solve_itinerary(
                    potential_hotels=hotels_list,
                    potential_attractions=attractions_list,
                    trip_duration_days=int(trip_days),
                    max_daily_hours=float(max_daily_hours),
                    is_daily_limit_flexible=bool(flexible),
                    objective_weights=objective_weights,
                )
            except Exception as e:
                st.exception(f"Error while running optimizer: {e}")
                itineraries = None

        if not itineraries:
            st.warning("No itineraries produced — check logs and API keys")
        else:
            st.success("Optimization finished")
            for it in itineraries:
                st.subheader(f"{it.get('title', 'Itinerary')} — total distance: {it.get('total_distance')}")
                for day in it.get("daily_routes", []):
                    with st.expander(f"Day {day.get('day')} — dist: {day.get('distance')} km"):
                        st.write(day.get("route"))
                        st.write({
                            "travel_time": day.get("travel_time"),
                            "visit_time": day.get("visit_time"),
                            "total_time": day.get("total_time"),
                        })

            pretty = json.dumps(itineraries, indent=2, ensure_ascii=False)
            st.download_button("Download results (JSON)", data=pretty, file_name="itineraries.json", mime="application/json")

st.markdown("---")
st.write("Notes: set st.secrets['GEOAPIFY_API_KEY'] and/or st.secrets['GOOGLE_API_KEY'] before running.\nrequirements: streamlit,pandas,requests,mip,googlemaps(optional)")
