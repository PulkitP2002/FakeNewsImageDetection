import streamlit as st
from PIL import Image
from ocr_caption import extract_text_from_image, generate_caption
from gan_checker import check_if_image_fake
from verifier import fetch_serp_news, check_similarity

st.set_page_config(page_title="🕵️‍♂️ Fake News Detector", layout="wide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("About This App")
    st.markdown("""
    This tool helps you detect if an **image is AI-generated** and verifies the **truthfulness of its content** using **Google News**.

    **Steps:**
    1. Upload an image
    2. Check if it's real or fake
    3. Extract or generate text
    4. Cross-check using Google News

    ---
    🔧 Built with: `Streamlit`, `PyTorch`, `EasyOCR`, `NewsAPI`
    """)
    st.markdown("Made by [Pulkit Prasad](mailto:pulkit.prasad22@gmail.com)")

# ---------------- Main Title ----------------
st.markdown("<h1 style='text-align: center;'>️Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and verify its authenticity & news content</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Upload Section ----------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.image(image, caption="Uploaded Image")

    # ---------------- Real/Fake Detection ----------------
    with right_col:
        st.markdown("### Step 1: Image Authenticity")
        img_result = check_if_image_fake(image)

        color = "green" if img_result == "Real" else "red"
        st.markdown(f"<div style='padding:10px; border-radius:10px; background-color:{color}; color:white; font-weight:bold;'>🧠 Detected as: {img_result}</div>", unsafe_allow_html=True)

        # ---------------- Extract Text ----------------
        st.markdown("### Step 2: Extract or Generate Information")
        extracted_text = extract_text_from_image(image)
        st.write("Extracted Text:", extracted_text)
        if extracted_text.strip() == "":
            st.warning("No text found in image! Trying caption generation...")
            extracted_text = generate_caption(image)
            st.info(f"Generated Caption: **{extracted_text}**")
        else:
            st.success("Text successfully extracted from image.")
            st.markdown(f"<div style='padding:10px; border:1px solid #ddd; border-radius:10px;'><strong>{extracted_text}</strong></div>", unsafe_allow_html=True)

        # ---------------- News Search & Verification ----------------
        st.markdown("### Step 3: Verifying Against NewsAPI")

        headlines = fetch_serp_news(extracted_text)
        st.write(f"Response: {headlines}")
    if not headlines:
        st.error("Could not fetch news articles.")
    else:
        st.write("Top Related Headlines:")
        for idx, h in enumerate(headlines, 1):
            st.markdown(f"**{idx}.** {h}")

        similarity = check_similarity(extracted_text, headlines)

        st.markdown("### Similarity Score")
        st.progress(similarity)
        st.markdown(f"<p style='font-size:18px'><strong>Similarity:</strong> {similarity * 100:.2f}%</p>", unsafe_allow_html=True)

        # ---------------- Final Verdict ----------------
        st.markdown("---")
        st.markdown("### Final Verdict")

        if similarity >= 0.5:
            st.success("This news appears to be **REAL** based on the matching headlines.")
        else:
            st.error("This news might be **FAKE** — not enough matching news found.")