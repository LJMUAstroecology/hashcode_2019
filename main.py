import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import random
import itertools

# From Python cookboook
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

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
    
    def add_photo(self, _photo):
        assert(photo not in self.photos)

        if len(self.photos) == 0:
            prev_slide = photo([], None)
        else:
            prev_slide = self.photos[-1]    
        
        next_slide = _photo

        self.score += transition_score(prev_slide, next_slide)
        
        self.photos.append(_photo)


def merge_vertical_photos(photos):
    pass

def generate_similarity_matrix(photos):

    similarity = np.zeros((len(photos), len(photos)), dtype='uint8')

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

    slideshows = []

    for subset in grouper(10, photos):
        best_score = 0

        for combo in itertools.combinations(subset, 10):
            s = slideshow()
            s.add_photos(combo)

            if s.score >= best_score:
                best_score = s.score
                best_s = s

        slideshows.append(best_s)

    return slideshows

if __name__ == "__main__":
    photos = parse_input(sys.argv[1])

    random.seed(42)
    random.shuffle(photos)

    photo_subset = photos[:100]
    similarity = generate_similarity_matrix(photo_subset)
    optimal_subsets(photo_subset)

    plt.imshow(similarity)
    plt.colorbar()
    plt.show()
