import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import random
import itertools

class photo:
    def __init__(self, tags, orientation):
        self.tags = set(tags)
        self.orientation = orientation

def transition_score(current_photo, next_photo):
    score_1 = len(current_photo.tags.union(next_photo.tags))
    score_2 = len(current_photo.tags.difference(next_photo.tags))
    score_3 = len(next_photo.tags.difference(current_photo.tags))

    return min(score_1, score_2, score_3)

class slideshow:
    def __init__(self):
        self.photos = []
        self.score = 0

    def add_photos(self, photos):
        for photo in photos:
            self.add_photo(photo)
    
    def add_photo(self, photo):
        assert(photo not in self.photos)

        prev_slide = self.photos[-1]    
        next_slide = photo

        self.score += transition_score(prev_slide, next_slide)
        self.photos.append(photo)


def merge_vertical_photos(photos):
    pass

def generate_similarity_matrix(photos):

    similarity = np.zeros((len(photos), len(photos)))

    for i, photo_i in tqdm(enumerate([p for p in photos if p.orientation == "H"])):
        for j, photo_j in tqdm(enumerate([p for p in photos if p.orientation == "H"])):
            similarity[i, j] = transition_score(photo_i, photo_j)

    return similarity

def parse_input(input_filename):
    with open(input_filename, 'r') as f:
        data = f.readlines()

    photos = []

    for meta in data[1:]:
        meta = meta.split(" ")

        orientation = meta[0]
        tags = meta[2:]

        photos.append(photo(tags, orientation))

    return photos

def optimal_subsets(photos, n=10):

    subsets = photos[::n]
    slideshows = []

    for subset in subsets:
        best_score = 0

        for combo in itertools.combinations(subset):
            s = slideshow()
            s.add_photos(combo)

            score = s.score_slideshow()

            if score >= best_score:
                best_score = score
                best_s = s

        slideshows.append(best_s)

    return slideshows

if __name__ == "__main__":
    photos = parse_input(sys.argv[1])

    random.seed(42)
    random.shuffle(photos)

    similarity = generate_similarity_matrix(photo_subset)

    plt.imshow(similarity)
    plt.colorbar()
    plt.show()
