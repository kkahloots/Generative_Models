def similarity_to_distance(similarity_fn):
    distance_fn = lambda x, y: 0.5*(1-similarity_fn(x,y))
    return distance_fn