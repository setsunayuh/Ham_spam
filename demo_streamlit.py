import streamlit as st

# Using menu
st.title("Trung Tâm Tin Học")

menu = ["Home", "Capstone Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""### Có 3 chủ đề trong khóa học:
    - Topic 1: RFM & Clustering
    - Topic 2: Recommendation System
    - Topic 3: Sentiment Analysis
    - ...""")
elif choice == "New Prediction":
    st.header("Please input to prediction")

