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

def transition_score(tags_a, tags_b):
    score_1 = len(tags_a.intersection(tags_b))
    score_2 = len(tags_a.difference(tags_b))
    score_3 = len(tags_b.difference(tags_a))

    return min(score_1, score_2, score_3)

class photo:
    def __init__(self, id, tags, orientation = "H"):
        self.tags = set(tags)
        self.index = id
        self.orientation = orientation

class slide:
    def __init__(self, photos):
        self.tags = set()
        self.photos = photos

        for p in self.photos:
            self.tags = self.tags.union(p.tags)

    def transition_score(self, next_slide):
        return transition_score(self.tags, next_slide.tags)

class slideshow:
    def __init__(self, slides = None):
        self.slides = []
        self.score = 0

        if slides is not None:
            self.add_slides(slides)
    
    def add_slides(self, slides):
        for s in slides:
            self.add_slide(s)
    
    def add_slide(self, next_slide):

        if len(next_slide.photos) == 0:
            return

        if len(self.slides) == 0:
            prev_slide = slide([])
        else:
            prev_slide = self.slides[-1]
        
        self.score += prev_slide.transition_score(next_slide)
        self.slides.append(next_slide)

    def save(self, filename):

        with open(filename, "w") as of:
            of.write("{}\n".format(len(self.slides)))
            for slide in self.slides:
                ids = []
                for photo in slide.photos:
                    ids.append(photo.index)
                
                of.write(" ".join([str(id) for id in ids]))
                of.write("\n")

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

    for i, meta in enumerate(data[1:]):
        meta = meta.split(" ")

        orientation = meta[0]
        tags = meta[2:]

        photos.append(photo(i, tags, orientation))

    return photos

def optimal_subsets(slides, n=5):

    slideshows = []

    for subset in tqdm(grouper(n, slides, slide([]))):

        best_score = 0

        for permutation in itertools.permutations(subset, n):
            s = slideshow()

            s.add_slides(permutation)

            if s.score >= best_score:
                best_score = s.score
                best_s = s

        slideshows.append(best_s)

    return slideshows

def plot_similarity_matrix(similarity):
    plt.imshow(similarity)
    plt.colorbar()
    plt.show()

def merge_slideshows(slideshows):

    master_show = slideshow()

    for s in slideshows:
        master_show.add_slides(s.slides)

    return master_show

def get_vertical_images():
    pass

if __name__ == "__main__":
    photos = parse_input(sys.argv[1])

    #random.seed(42)
    #random.shuffle(photos)

    photos = photos
    slides = [slide([photo]) for photo in photos]

    slideshows = optimal_subsets(slides, 6)

    # Calculate reference slideshow score
    ref_slideshow = slideshow(slides)
    print("Reference (random) slideshow score: ", ref_slideshow.score)

    combined = merge_slideshows(slideshows)

    print("Combined score: ", combined.score)

    combined.save("output.txt")

