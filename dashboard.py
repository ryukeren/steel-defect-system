import streamlit as st
import requests
from PIL import Image
import base64
import io
import pandas as pd

API_URL = "http://api:8000"

st.set_page_config(page_title="Steel Defect Monitoring System", layout="wide")

st.title("SMART STEEL DEFECT SYSTEM")

st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload Steel Surface Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------
# Batch Detection Section
# -------------------------

if uploaded_files:

    if st.button("Run Batch Defect Detection"):

        results = []

        for uploaded_file in uploaded_files:

            st.markdown("---")
            st.subheader(uploaded_file.name)

            image = Image.open(uploaded_file)

            col1, col2, col3 = st.columns(3)

            # Input Image
            with col1:
                st.text("Input Image")
                st.image(image, width="stretch")

            files = {"file": uploaded_file.getvalue()}

            try:

                response = requests.post(f"{API_URL}/predict", files=files)
                data = response.json()

                if data.get("status") == "success":

                    results.append({
                        "Image": uploaded_file.name,
                        "Defects": data.get("total_detections", 0),
                        "Processing Time (ms)": data.get("processing_time_ms", 0)
                    })

                    # Annotated Detection Image
                    img_bytes = base64.b64decode(data["annotated_image_base64"])
                    result_image = Image.open(io.BytesIO(img_bytes))

                    with col2:
                        st.text("Detection Output")
                        st.image(result_image, width="stretch")

                    # Heatmap
                    heatmap_bytes = base64.b64decode(data["heatmap_image_base64"])
                    heatmap_image = Image.open(io.BytesIO(heatmap_bytes))

                    with col3:
                        st.text("Defect Heatmap")
                        st.image(heatmap_image, width="stretch")

                else:
                    st.error(f"Prediction failed for {uploaded_file.name}")

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        df = pd.DataFrame(results)

        st.markdown("---")
        st.subheader("Batch Inspection Summary")

        st.dataframe(df)

        st.subheader("Defect Distribution")

        st.bar_chart(df.set_index("Image")["Defects"])

# -------------------------
# Analytics Section
# -------------------------

st.markdown("---")

st.subheader("System Analytics")

if st.button("Load Analytics"):

    try:
        response = requests.get(f"{API_URL}/analytics")
        analytics = response.json()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Runs", analytics.get("runs", 0))
            st.metric("Total Detections", analytics.get("total_detections", 0))

        with col2:
            severity_dist = analytics.get("severity_distribution", {})

            if severity_dist:
                df = pd.DataFrame(
                    list(severity_dist.items()),
                    columns=["Severity", "Count"]
                )

                st.bar_chart(df.set_index("Severity"))

    except Exception as e:
        st.error(f"Error fetching analytics: {e}")