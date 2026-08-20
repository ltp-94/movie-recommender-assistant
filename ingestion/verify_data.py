from elasticsearch import Elasticsearch
from config import INDEX_NAME, ELASTICSEARCH_URL

es = Elasticsearch(ELASTICSEARCH_URL)

# Just pull the first movie to see if it looks correct
res = es.search(index=INDEX_NAME, body={"query": {"match_all": {}}, "size": 1})
movie = res['hits']['hits'][0]['_source']

print(f"Title: {movie['title']}")
print(f"Rating: {movie['vote_average']}")
print(f"Poster: {movie['poster_url']}")
print(f"Link:   {movie['movie_link']}")