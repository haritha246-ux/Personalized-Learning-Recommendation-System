import streamlit as st
import pickle

similarity, df = pickle.load(
    open("models/similarity_model.pkl", "rb")
)

st.title("Coursera Course Recommendation System")

course = st.selectbox(
    "Select a Course",
    df["course_title"].values
)

if st.button("Recommend Courses"):

    idx = df[df["course_title"] == course].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("Recommended Courses")

    for i in scores:
        st.write(df.iloc[i[0]]["course_title"])