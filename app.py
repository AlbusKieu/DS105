import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load necessary files
with open('user_item_matrix.pkl', 'rb') as file:
    user_item_matrix = pickle.load(file)

new_df = pd.read_csv('new_df.csv')
user_data = pd.read_csv('user_data.csv')
movie_data = pd.read_csv('movie_data.csv')  # File containing movie metadata
movies = new_df[['ID', 'Movie_Name']]

# Calculate user similarity
user_similarity = cosine_similarity(user_item_matrix)

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(new_df['Tags'].tolist())

# Calculate content-based similarity
similarity = cosine_similarity(embeddings)

def hybrid_recommender_balanced(target_user_id, movie_name, new_df = new_df, user_item_matrix=user_item_matrix, user_similarity=user_similarity, similarity=similarity, alpha=0.5, top_n=5):
    target_user_idx = target_user_id - 1
    user_user_similarity_vector = user_similarity[target_user_idx]
    target_user_ratings = user_item_matrix.iloc[target_user_idx]
    rated_movies = target_user_ratings[target_user_ratings > 0].index
    total_movies = movies.shape[0]
    user_based_recommendations = []
    for movie_idx in range(1, total_movies + 1):
        if movie_idx in rated_movies:
            user_based_recommendations.append(0.0)
            continue
        else:
            numerator = 0
            denominator = 0
            other_rated_users = user_item_matrix[user_item_matrix[movie_idx] > 0].index
            for other_user in other_rated_users:
                other_user_idx = other_user - 1
                similarity_score = user_user_similarity_vector[other_user_idx]
                numerator += similarity_score * user_item_matrix.loc[other_user, movie_idx]
                denominator += abs(similarity_score)
            if denominator > 0:
                predicted_rating = numerator / denominator
                user_based_recommendations.append(predicted_rating)
            else:
                user_based_recommendations.append(0.0)
    ub_predicted_vector = np.array(user_based_recommendations)
    # Chuẩn hóa lại về vùng 0 - 1 cho tương đồng với content-based
    # vì content-based sử dụng cosin
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(ub_predicted_vector.reshape(-1, 1))
    normalized_array = normalized_array.flatten()

    # Phần 2: đã tính similarity

    # Phần 3
    # Tương đồng của user + tương đồng của phim = kết hợp giữa 2 vùng value
    movie_idx = new_df[new_df['Movie_Name'] == movie_name].index[0]
    # Chỉ khi dùng cos sim với chính nó thì cho ra kết quả là 1, do đó
    # cho kết quả = 0 khi sử dụng cos sim với movie đó
    movie_vector = similarity[movie_idx]
    movie_vector[movie_idx] = 0.0
    hybrid_array = alpha * normalized_array + (1 - alpha) * similarity[movie_idx]
    hybrid_dict = {}
    for movie_idx in range(1, total_movies + 1):
        hybrid_dict[movie_idx] = float(hybrid_array[movie_idx - 1])
    recommendations = sorted(hybrid_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations

# Streamlit Interface
st.markdown("<h1 style='text-align: center; color: #d35400;'>Let’s Find the Perfect Movie that Matches Your Vibe!🎬</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Just pick a title and let us do the magic ✨</p>", unsafe_allow_html=True)
# Input for user ID and movie name
target_user_id = st.number_input("Enter User ID:", min_value=1, step=1)
movie_name = st.selectbox("Select Movie Name:", options=new_df['Movie_Name'].tolist())

st.markdown("""
    <style>
    .recommend-btn {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        font-size: 18px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        cursor: pointer;
    }
    .recommend-btn:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("Get Recommendations"):
    with st.spinner("Please wait a moment as we discover your next favorite movies! 🍿"):

        if target_user_id and movie_name:
            try:
                recommendations = hybrid_recommender_balanced(target_user_id, movie_name)

                for movie_id, predicted_rating in recommendations:
                    movie_name_recommended = movies.loc[movies['ID'] == movie_id, 'Movie_Name'].values[0]
                    movie_info = movie_data[movie_data['Movie_Name'] == movie_name_recommended].iloc[0]
                    poster_url = movie_info['Poster Links']
                    st.image(poster_url, width=150, caption=movie_name_recommended)

                    st.write(f"**Predicted Score:** {predicted_rating:.2f}")

                    with st.expander(f"Details for {movie_name_recommended}"):
                        for column in movie_data.columns:
                            if column != 'Poster Links':
                                st.write(f"**{column}:** {movie_info[column]}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("Please provide both User ID and Movie Name.")
