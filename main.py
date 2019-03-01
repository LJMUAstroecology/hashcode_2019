import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import random
import itertools
import os
import time

from photo import Photo, transition_score
from slide import Slide
from slideshow import Slideshow

# From Python cookboook
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

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
        meta = meta.strip().split(" ")

        orientation = meta[0]
        tags = meta[2:]

        photos.append(Photo(i, tags, orientation))

    return photos

def optimal_subsets(slides, n=5):

    slideshows = []

    for subset in tqdm(grouper(n, slides, Slide([]))):

        best_score = 0

        for permutation in itertools.permutations(subset, n):
            s = Slideshow()

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

    merged_slideshow = Slideshow()

    for slideshow in slideshows:
        merged_slideshow.add_slides(slideshow.slides)

    return merged_slideshow

def find_greedy_match_slide(slide, slides):
    """
    Simple strategy for finding the best match for a given (horizontal!) photo.

    Complexity is O(N) where N = len(photos)

    Returns the *index* of the best match
    """

    if len(slides) == 1:
        return 0

    best_score = 0
    best_match = random.randint(0, len(slides)-1)

    for i, s in enumerate(slides):
        score = slide.score(s)
        
        if score > best_score:
            best_score = score
            best_match = i
    
    return best_match

def find_greedy_match_vertical(photo, photos):
    """
    Simple strategy for finding the best matching vertical photo.

    Finds combinations of vertical photos with high tag differences.
    """

    if len(photos) == 1:
        return 0

    best_score = 0
    best_match = random.randint(0, len(photos)-1)

    for i, p in enumerate(photos):
        score = len(photo.tags.union(p.tags))

        if score > best_score:
            best_score = score
            best_match = i

    return best_match

def get_horizontal_slides(photos):
    return [Slide([p]) for p in photos if p.orientation == "H"]

def get_vertical_slides(photos, strategy="naive"):

    slides = []
    verticals =  [p for p in photos if p.orientation == "V"]

    if strategy == "naive":
        print("Naive verticals")
        for i in range(0, len(verticals), 2):
            print(i, i+1)
            slides.append(Slide( [verticals[i], verticals[i+1]] ))
    elif strategy == "greedy":
        print("Greedy verticals")
        for _ in verticals:
            v = verticals.pop()
            best_match = find_greedy_match_vertical(v, verticals)
            slides.append(Slide( [v, verticals.pop(best_match)]))
            
    return slides

def get_slides(photos, strategy="naive"):
    return get_horizontal_slides(photos) + get_vertical_slides(photos, strategy)

def greedy_slideshow(photos, rand_seed=42, strategy="naive"):
    slides = get_slides(photos, strategy)

    random.seed(rand_seed)
    random.shuffle(slides)

    print("{} slides".format(len(slides)))

    show = Slideshow([slides.pop()])

    n_slides = len(slides)
    pbar = tqdm(total=n_slides)

    # This runs for N^2 iterations...

    pbar = tqdm(range(n_slides))
    for _ in pbar:
        best_slide = find_greedy_match_slide(show.slides[-1], slides)
        show.add_slide(slides.pop(best_slide))

        pbar.set_postfix(score=show.score, n_slides=len(show.slides))

    return show


def optimal_subset_slideshow(photos):
    slides = get_horizontal_slides(photos) + get_vertical_slides(photos)
    slideshows = optimal_subsets(slides, 6)
    return  merge_slideshows(slideshows)

if __name__ == "__main__":

    input_file = sys.argv[1]
    photos = parse_input(input_file)

    # Calculate reference slideshow score
    #ref_slideshow = Slideshow(slides)
    #print("Reference (random) slideshow score: ", ref_slideshow.score)

    show = greedy_slideshow(photos, rand_seed=time.time(), strategy="naive")

    print("Slideshow score: ", show.score)

    show.save("test_greedy_{}.txt".format(os.path.splitext(os.path.basename(input_file))[0]))
