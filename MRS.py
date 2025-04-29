# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#we have used tmdb dataset
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies.head()

credits.head()

"""# **DATA PREPROCESSING**"""

movies = movies.merge(credits,on='title')

movies.head()

movies=movies[['movie_id','title','genres','overview','keywords','cast','crew']]
movies.head()

movies.isnull().sum()

movies.dropna(inplace=True)
movies.isnull().sum()

movies.duplicated().sum()

m=movies.iloc[0].genres
#it is list of dictionaries so convert into [action, adventure, fantasy, scifi]

#to convert string into list
import ast
#m=ast.literal_eval(m)

def convert(obj):
  c=[]
  for i in ast.literal_eval(obj):
    c.append(i['name'])
  return c

movies['genres']=movies['genres'].apply(convert)

movies.head()

movies['keywords']=movies['keywords'].apply(convert)

movies.head()

movies['cast'][0]
#extract 4 names of cast from cast

def convertcast(obj):
  c=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter !=4:
      c.append(i['name'])
      counter+=1
    else:
      break
  return c

movies['cast']=movies['cast'].apply(convertcast)

movies.head()

movies['crew'][0]
#extract director and name from crew

def extract(obj):
  c=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      c.append(i['name'])
      break
  return c

movies['crew']=movies['crew'].apply(extract)

movies.head()

movies['overview'][0]
#it is a string convert into list

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

#concatenate lists, and then merge those list into string which will output in tags

#remove spaces between names and words b/c every word will become different tag however johnny depp is one word but sue to space it will divide into 2 other tags which will create confusion for model
#for eg science fiction is one word but it will divide science into 1 tag and fiction in other

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])

movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])

movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

movies.head()

"""##**EDA**"""

# @title movie_id

from matplotlib import pyplot as plt
movies['movie_id'].plot(kind='line', figsize=(8, 4), title='movie_id')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title movie_id

from matplotlib import pyplot as plt
movies['movie_id'].plot(kind='hist', bins=20, title='movie_id')
plt.gca().spines[['top', 'right',]].set_visible(False)

#no need of other columns so remove other columns and only tags is needed

newdf = movies[['movie_id','title','tags']]
newdf

newdf['tags']=newdf['tags'].apply(lambda x:" ".join(x))

newdf

newdf['tags']=newdf['tags'].apply(lambda x:x.lower())

newdf.head()

"""#**VECTORIZATION**"""

newdf['tags'][0]

newdf['tags'][1]

#convert text into vectors and calculate similarity between tags
#check closest vectors for recommendation
#bag of words -> technique to convert text into vectors
#countvectorizer to remove stop words

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(newdf['tags']).toarray()

vector.shape

vector

print(cv.get_feature_names_out())

len(cv.get_feature_names_out())

#apply stemming to your text
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

#example:
stem('learning is better')

stem(newdf['tags'][0])

newdf['tags']=newdf['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(newdf['tags']).toarray()

vector

cv.get_feature_names_out()

#calculate cosine distance since euclidean is not good for higher dimension
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(vector)

sim.shape

ls = list(enumerate(sim[0])) #similarity of first movie with all movies

sorted(ls,reverse=True,key=lambda x:x[1])[1:6]

"""##**CONTENT BASED SAMPLE CODE**"""

def recommend_content2(movie):
  movie_index = newdf[newdf['title']==movie].index[0]
  distance = sim[movie_index]
  m_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

  for i in m_list:
    print(newdf.iloc[i[0]].title)

recommend_content2('Avatar')

recommend_content2('Batman Begins')

"""##**content based function using n recommendation**"""

def recommend_content(movie, sim, newdf, n_recommendations=10):
    movie_index = newdf[newdf['title'] == movie].index[0]
    distance = sim[movie_index]
    m_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1: n_recommendations]

    recommended_movies = []
    for i in m_list:
        recommended_movies.append(newdf.iloc[i[0]].title)

    print(f"Recommended movies for '{movie}':")
    for movie in recommended_movies:
        print(movie)
    return recommended_movies

recommend_content("Avatar", sim, newdf)

"""##**COLLABOATIVE FILTERING**:"""

m=pd.read_csv('tmdb_5000_movies.csv')
m

"""##**POPULAR MOVIES**"""

most_popular_movies = m.sort_values(by='popularity', ascending=False)
top_n = 10
top_popular_movies = most_popular_movies[['title', 'popularity', 'vote_count', 'vote_average']].head(top_n)
print("Top Popular Movies:")
print(top_popular_movies)

m=m[['id','title','genres','vote_count', 'popularity','vote_average']]
m.head()

m.isnull().sum()

m.duplicated().sum()

#to convert string into list
import ast
#m=ast.literal_eval(m)

def convert(obj):
  c=[]
  for i in ast.literal_eval(obj):
    c.append(i['name'])
  return c

m['genres']=m['genres'].apply(convert)

m

m.describe()

"""##**WEIGHTED SCORE**"""

vote_count_weight = 1.0
popularity_weight = 0.5
m['weighted_score'] = (m['vote_average'] * m['vote_count'] * vote_count_weight) + (m['popularity'] * popularity_weight)
sorted_movies = m.sort_values(by='weighted_score', ascending=False)
print(sorted_movies[['title', 'weighted_score']].head(10))

m

import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(m['id'].unique())}
rows = [movie_id_to_index[movie_id] for movie_id in m['id']]
cols = rows
values = m['weighted_score']

user_item_matrix = csr_matrix((values, (rows, cols)), shape=(len(movie_id_to_index),len(movie_id_to_index)))
#model
model = AlternatingLeastSquares(factors=100, regularization=0.05, iterations=30)
model.fit(user_item_matrix.T)

print(user_item_matrix)
print(f"Shape: {user_item_matrix.shape}, Non-zero values: {user_item_matrix.nnz}")

"""##**collaborative filtering**"""

def get_movie_id_by_name(movie_name, movies_df):
    movie_row = movies_df[movies_df['title'].str.contains(movie_name, case=False, na=False)]

    if not movie_row.empty:
        movie_id = movie_row.iloc[0]['id']
        return movie_id
    else:
        return None

def recommend_collaborative(movie, n_recommendations=10):
    movie_id = get_movie_id_by_name(movie, m)
    if movie_id is None:
        return f"Movie '{movie}' not found in the dataset."

    #internal index of the movie
    movie_index = m[m['id'] == movie_id].index[0]
    print(f"Movie ID: {movie_id}, Movie Index: {movie_index}, Title: {m.iloc[movie_index]['title']}")

    recommendations = model.similar_items(movie_index, N=n_recommendations + 1)
    recommended_movies = []
    for i in range(1, len(recommendations[0])):  # Skip the first recommendation
        als_movie_index = recommendations[0][i]  # Extract the movie index from the recommendations
        als_movie_index = int(als_movie_index)

        if 0 <= als_movie_index < len(m):
            print(f"Recommendation Index: {als_movie_index}, Title: {m.iloc[als_movie_index]['title']}")
            movie_title = m.iloc[als_movie_index]['title']
            recommended_movies.append(movie_title)
        else:
            print(f"Invalid Index: {als_movie_index}")
    return recommended_movies

# Example usage
movie = 'Batman'
recommended_movies = recommend_collaborative(movie)
print("Recommended Movies:", recommended_movies)

movie = 'Avatar'
recommended_movies = recommend_collaborative(movie)
print("Recommended Movies:", recommended_movies)

"""##**Filtering on Genre**"""

def filtered(movie):
  input_movie_genres = set(m[m['title'] == movie]['genres'].iloc[0])
  recommended_movies = recommend_collaborative(movie)
  filtered_recommendations = []
  for movie in recommended_movies:
      movie_genres = set(m[m['title'] == movie]['genres'].iloc[0])
      if input_movie_genres & movie_genres:
          filtered_recommendations.append(movie)

  return filtered_recommendations

filter=filtered("Avatar")
print("FILTERED RECOMMENDATIONS ARE: ",filter)

"""##**HYBRID MODEL(content and collaborative)**"""

def hybrid_recommendation(movie, n_recommendations=10, collaborative_weight=0.4, content_weight=0.6):
    print("collaborative:")
    collaborative_recs = filtered(movie)
    print("content:")
    content_recs = recommend_content(movie, sim, newdf)

    num_collab_recs = int(collaborative_weight * n_recommendations)
    num_content_recs = int(content_weight * n_recommendations)

    all_recommendations = list(set(collaborative_recs[:num_collab_recs] + content_recs[:num_content_recs]))

    return all_recommendations[:n_recommendations]

movie = "Avatar"
recommended_movies = hybrid_recommendation(movie)
print("Hybrid Recommended Movies:", recommended_movies)

"""##**ACCURACY**"""

newdf.head()

movie='Avatar'
relevant_movies_collaborative = filtered(movie)

print(f"Relevant Movies for Avatar (Collaborative Filtering): {relevant_movies_collaborative}")

from sklearn.metrics import precision_score, recall_score, f1_score
relevant_movies = relevant_movies_collaborative
recommended_movies

y_true = [1 if movie in relevant_movies else 0 for movie in recommended_movies]  # Ground truth
y_pred = [1 if movie in relevant_movies else 0 for movie in recommended_movies]  # Predicted by model

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

from sklearn.metrics import precision_score, recall_score, f1_score

relevant_movies = set(['Titan A.E.', 'Independence Day', 'Aliens vs Predator: Requiem', 'Predators', 'Jupiter Ascending'])
recommended_movies

y_true = [1 if movie in relevant_movies else 0 for movie in recommended_movies]  # Ground truth
y_pred = [1 if movie in relevant_movies else 0 for movie in recommended_movies]  # Predicted by model

# Precision, Recall, and F1 Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

from sklearn.metrics import precision_score, recall_score, f1_score

# Example ground truth (relevant movies), these could be user-provided or based on previous ratings
relevant_movies = set(['Titan A.E.', 'Independence Day', 'Aliens vs Predator: Requiem', 'Predators', 'Jupiter Ascending'])


# Generate relevant recommendations (intersection of recommended movies and ground truth)
relevant_recommendations = set(recommended_movies).intersection(relevant_movies)

# Precision, Recall, and F1-Score
def evaluate_recommendations(recommended, relevant):
    # Convert the set to a binary vector (1 if recommended, 0 otherwise)
    recommended_set = set(recommended)
    relevant_set = set(relevant)

    # Create binary vectors for precision, recall calculation
    y_true = [1 if movie in relevant_set else 0 for movie in recommended]
    y_pred = [1 if movie in recommended_set else 0 for movie in recommended]

    # Precision, Recall, F1-Score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

# Calculate metrics
precision, recall, f1 = evaluate_recommendations(relevant_recommendations, relevant_movies)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

# Example ground truth (relevant movies)
relevant_movies = set(['Titan A.E.', 'Independence Day', 'Aliens vs Predator: Requiem', 'Predators', 'Jupiter Ascending'])

def evaluate_recommendations(recommended, relevant):
    recommended_set = set(recommended)
    relevant_set = set(relevant)

    y_true = [1 if movie in relevant_set else 0 for movie in recommended]
    y_pred = [1 if movie in recommended_set else 0 for movie in recommended]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

precision, recall, f1 = evaluate_recommendations(recommended_movies, relevant_movies)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")