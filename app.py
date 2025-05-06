import streamlit as st
from PIL import Image
from utils.face_matcher import compare_faces

st.set_page_config(page_title="KYC Face Verification", layout="centered")
st.title("KYC Face Verification")
st.write("Upload a selfie and an ID image. The system will check if the faces match.")

# Upload inputs
selfie = st.file_uploader("Selfie Image", type=["jpg", "jpeg", "png"])
id_image = st.file_uploader("ID Card Image", type=["jpg", "jpeg", "png"])

# Show images once both are uploaded
if selfie and id_image:
    st.write("Preview of Uploaded Images:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(selfie, caption="Selfie", use_column_width=True)
    with col2:
        st.image(id_image, caption="ID Image", use_column_width=True)

    # Action button
    if st.button("Compare Faces"):
        result = compare_faces(selfie, id_image)
    
        if result[0]:
            st.success(f"Faces match with score: {result[1]}")
        else:
            st.error("Faces do not match.")
