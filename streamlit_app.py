import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# ---------- Load saved model, scaler, and data ----------
model = joblib.load('nn_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('cleaned_songs.csv')

# ---------- Rebuild features, scaling, and song index ----------
features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key',
            'loudness', 'mode', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']

df_scaled = scaler.transform(df[features])

song_index = pd.Series(df.index, index=df['track_name'])
song_index = song_index[~song_index.index.duplicated(keep='first')]

# ---------- Recommendation function (song name only, genre-aware) ----------
def recommend_songs(song_name, num_recommendations=5, same_genre_only=True):
    if song_name not in song_index.index:
        matches = df[df['track_name'].str.contains(song_name, case=False, na=False)]
        raise ValueError(f"'{song_name}' not found. Did you mean: {list(matches['track_name'][:5])}")

    idx = song_index[song_name]
    song_genre = df.loc[idx, 'track_genre']

    if same_genre_only:
        candidates = df[df['track_genre'] == song_genre]
    else:
        candidates = df

    candidate_scaled = df_scaled[candidates.index]
    temp_model_input = df_scaled[idx].reshape(1, -1)

    temp_model = NearestNeighbors(n_neighbors=min(num_recommendations + 1, len(candidates)), metric='cosine')
    temp_model.fit(candidate_scaled)
    distances, indices = temp_model.kneighbors(temp_model_input)

    recommended_positions = indices[0][1:]
    recommended_indices = candidates.index[recommended_positions]
    return df.loc[recommended_indices][['track_name', 'artists', 'track_genre', 'popularity', 'danceability', 'energy']]

# ---------- Recommendation function (song name + artist, genre-aware) ----------
def recommend_songs_by_artist(song_name, artist_name, num_recommendations=5, same_genre_only=True):
    match = df[(df['track_name'].str.lower() == song_name.lower()) &
               (df['artists'].str.contains(artist_name, case=False, na=False))]
    if match.empty:
        raise ValueError(f"No match found for '{song_name}' by '{artist_name}'")

    idx = match.index[0]
    song_genre = df.loc[idx, 'track_genre']

    if same_genre_only:
        candidates = df[df['track_genre'] == song_genre]
    else:
        candidates = df

    candidate_scaled = df_scaled[candidates.index]
    temp_model_input = df_scaled[idx].reshape(1, -1)

    temp_model = NearestNeighbors(n_neighbors=min(num_recommendations + 1, len(candidates)), metric='cosine')
    temp_model.fit(candidate_scaled)
    distances, indices = temp_model.kneighbors(temp_model_input)

    recommended_positions = indices[0][1:]
    recommended_indices = candidates.index[recommended_positions]
    return df.loc[recommended_indices][['track_name', 'artists', 'track_genre', 'popularity', 'danceability', 'energy']]

# ---------- Fuzzy search helper ----------
def search_song_options(query, limit=10):
    if not query:
        return []
    matches = df[df['track_name'].str.contains(query, case=False, na=False)]
    options = (matches['track_name'] + " — " + matches['artists']).unique()
    return list(options[:limit])

# ---------- Streamlit UI ----------
st.title("🎵 Song Recommender")
st.caption("Find songs similar to your favorites, based on audio features and genre.")
st.divider()

search_mode = st.radio("Search by:", ["Song name only", "Song name + Artist"])
num_recs = st.slider("Number of recommendations:", min_value=3, max_value=15, value=5)
same_genre = st.checkbox("Only recommend songs from the same genre", value=True)

if search_mode == "Song name only":
    query = st.text_input("Type a song name to search:", key="query_only")
    options = search_song_options(query)
    if options:
        selected = st.selectbox("Select the exact song:", options, key="select_only")
        song = selected.split(" — ")[0]
        try:
            results = recommend_songs(song, num_recommendations=num_recs, same_genre_only=same_genre)
            st.subheader(f"Recommendations for '{song}'")
            st.dataframe(results, use_container_width=True)
        except ValueError as e:
            st.error(str(e))
    elif query:
        st.warning("No matches found. Try a different search term.")

else:
    query = st.text_input("Type a song name to search:", key="query_artist")
    options = search_song_options(query)
    if options:
        selected = st.selectbox("Select the exact song:", options, key="select_artist")
        song, artist = selected.split(" — ")
        try:
            results = recommend_songs_by_artist(song, artist, num_recommendations=num_recs, same_genre_only=same_genre)
            st.subheader(f"Recommendations for '{song}' by '{artist}'")
            st.dataframe(results, use_container_width=True)
        except ValueError as e:
            st.error(str(e))
    elif query:
        st.warning("No matches found. Try a different search term.")