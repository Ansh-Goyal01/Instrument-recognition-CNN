import streamlit as st
import tempfile
import os
import pandas as pd

from src.inference import predict_instrument

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Instrument Recognition",
    page_icon="🎵",
    layout="wide"
)

# =========================
# CUSTOM DARK THEME
# =========================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #38bdf8;
}
.card {
    background-color:#020617;
    padding:20px;
    border-radius:15px;
    box-shadow:0 0 15px rgba(56,189,248,0.3);
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🎵 Instrument Recognition System")
st.write("Upload an audio file and let the CNN predict the instrument.")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1.2])

# =========================
# LEFT COLUMN – UPLOAD
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎧 Upload Audio")

    uploaded_file = st.file_uploader(
        "Choose a WAV or MP3 file",
        type=["wav", "mp3"]
    )

    if uploaded_file:
        st.audio(uploaded_file)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION SECTION
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction")

    if uploaded_file and st.button("🎯 Predict Instrument", use_container_width=True):
        with st.spinner("🎶 Listening carefully..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp.write(uploaded_file.read())
                temp_path = temp.name

            results = predict_instrument(temp_path)
            os.remove(temp_path)

        # =========================
        # MAIN RESULT
        # =========================
        best = results[0]

        # =========================
        # RELATIVE CONFIDENCE (Method 1)
        # =========================
        total_conf = sum(r["confidence"] for r in results)
        if total_conf > 0:
            relative_conf = round((best["confidence"] / total_conf) * 100, 1)
        else:
            relative_conf = 0.0

        st.success("Prediction Complete!")

        st.metric(
            label="🎼 Predicted Instrument",
            value=best["instrument"]
        )

        st.progress(relative_conf / 100)
        st.write(f"**Confidence (relative):** {relative_conf} %")

        st.caption(
            "Confidence is normalized across the top predictions for better interpretability."
        )

        # =========================
        # TOP 3 RESULTS
        # =========================
        st.subheader("🥇 Top 3 Predictions")

        df = pd.DataFrame(results)
        st.dataframe(
            df.style.highlight_max(axis=0, color="#14532d"),
            use_container_width=True
        )

        # Fire balloons once
        if "predicted" not in st.session_state:
            st.session_state.predicted = True
            st.balloons()

    else:
        st.info("Upload an audio file and click Predict.")

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
st.caption("🎵 Instrument Recognition using CNN | Built with PyTorch & Streamlit")
