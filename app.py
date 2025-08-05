import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Extended movie dataset with more movies, including Telugu movies
data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'title': ['The Matrix', 'John Wick', 'Toy Story', 'Finding Nemo', 'The Lion King',
              'The Dark Knight', 'Inception', 'Shrek', 'Frozen', 'The Avengers',
              'Avatar', 'Jurassic Park', 'Pirates of the Caribbean', 'Star Wars', 'The Godfather',
              'Baahubali: The Beginning', 'Baahubali: The Conclusion', 'RRR', 'Arjun Reddy', 'Eega'],
    'genres': [['Action', 'Sci-Fi'], ['Action', 'Thriller'], ['Animation', 'Comedy'],
               ['Animation', 'Adventure'], ['Animation', 'Drama'],
               ['Action', 'Crime', 'Drama'], ['Action', 'Sci-Fi', 'Thriller'],
               ['Animation', 'Comedy'], ['Animation', 'Family', 'Fantasy'],
               ['Action', 'Adventure', 'Sci-Fi'],
               ['Action', 'Sci-Fi', 'Adventure'], ['Action', 'Adventure', 'Sci-Fi'],
               ['Adventure', 'Fantasy', 'Action'], ['Action', 'Adventure', 'Fantasy'],
               ['Crime', 'Drama', 'Thriller'],
               ['Action', 'Drama', 'Fantasy'], ['Action', 'Drama', 'Fantasy'],
               ['Action', 'Drama', 'Thriller'], ['Romance', 'Drama', 'Thriller'],
               ['Fantasy', 'Action', 'Drama']],
}

df = pd.DataFrame(data)

# One-hot encode genres
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genres'])

# Convert to DataFrame
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df_encoded = pd.concat([df[['movie_id', 'title']], genre_df], axis=1)

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(genre_encoded)

# Recommendation function
def recommend(movie_title, df, sim_matrix):
    if movie_title not in df['title'].values:
        return "Movie not found in database."
    
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3 recommendations
    recommended_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[recommended_indices].tolist()

# Ask user for movie title input
user_input = input("Enter a movie title to get recommendations: ")

# Provide recommendations
recommendations = recommend(user_input, df, cosine_sim)

# Display the recommendations
print(f"Recommendations for '{user_input}':")
print(recommendations)


