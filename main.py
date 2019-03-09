import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import random
import itertools
import os
import time
import math

from photo import Photo, transition_score
from slide import Slide
from slideshow import Slideshow

input_file = sys.argv[1]

def generate_similarity_matrix(photos):
    """
    Convenience function to generate an adjacency matrix between all photos
    """

    similarity = np.zeros((len(photos),len(photos)), dtype='uint8')

    n = len(photos)

    for i, photo_i in tqdm(enumerate(photos)):
        for j, photo_j in enumerate(photos[i:]):

            similarity[i, j] = transition_score(photo_i.tags, photo_j.tags)

    return similarity


def plot_similarity_matrix(similarity):
    plt.imshow(similarity)
    plt.colorbar()
    plt.show()

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

def find_greedy_match_slide(slide, slides):
    """
    Simple strategy for finding the best match for a given (horizontal!) photo.

    Complexity is O(N) where N = len(photos)

    Returns the *index* of the best match
    """

    if len(slides) == 1:
        return 0

    best_possible = len(slide.tags)
    best_score = 0
    best_match = random.randint(0, len(slides)-1)

    for i, s in enumerate(slides):
        score = slide.score(s)
        
        if score > best_score:
            best_score = score
            best_match = i

            if score == best_possible:
                break
    
    return best_match

def find_greedy_match_photo(slide, photos):
    """
    Find the best photo
    """
    
    if len(photos) == 0:
        return 0, -1
        
    best_score = 0
    best_match = random.randint(0, len(photos)-1)
    
    comparison_slide = Slide()

    for i, p in enumerate(photos):
        comparison_slide.photos = [p]
        comparison_slide.tags = p.tags

        score = slide.score(comparison_slide)

        if score > best_score:
            best_score = score
            best_match = i

    return best_match, best_score

def find_greedy_match_vertical_photo(slide, photo, photos):
    """
    Find the best vertical photo
    """

    if len(photos) == 0:
        return 0, -1
        
    best_possible = len(slide.tags)
    best_score = 0
    best_match = random.randint(0, len(photos)-1)

    for i, p in enumerate(photos):

        score = slide.score(Slide(photo, p))

        if score > best_score:
            best_score = score
            best_match = i

            if score == best_possible:
                break
        
    return best_match, best_score

def get_horizontal_slides(photos):
    return [Slide(p) for p in photos if p.orientation == "H"]

def get_vertical_slides(photos, rand_seed=time.time()):

    random.seed(rand_seed)
    random.shuffle(photos)

    slides = []
    verticals = [p for p in photos if p.orientation == "V"]

    while len(verticals) > 0:
        slides.append(Slide(verticals.pop(), verticals.pop()))
            
    return slides

def get_slides(photos):
    return get_horizontal_slides(photos) + get_vertical_slides(photos)

def greedy_slideshow_slides(slides, rand_seed=time.time()):
    random.seed(rand_seed)
    random.shuffle(photos)

    show = Slideshow([slides.pop()])
    total_slides = len(slides)

    # This runs for N^2 iterations...
    pbar = tqdm(range(total_slides))
    for _, _ in enumerate(pbar):

        if len(slides) == 0:
            break

        best_idx = find_greedy_match_slide(show.slides[-1], slides)
        show.add_slide(slides.pop(best_idx))

        avg_score = show.score/len(show.slides)
        pbar.set_postfix(score=show.score, avg_score=avg_score)
    
    return show

def greedy_slideshow(photos, rand_seed=time.time()):

    random.seed(rand_seed)
    random.shuffle(photos)

    # Set up slideshow
    vertical_photos = [p for p in photos if p.orientation == "V"]
    horizontal_photos = [p for p in photos if p.orientation == "H"]

    n_h = len(horizontal_photos)
    n_v = len(vertical_photos)

    print(n_h, n_v)

    if n_h == 0:
        show = Slideshow([ Slide(vertical_photos.pop(), vertical_photos.pop()) ])
    else:
        show = Slideshow([ Slide(horizontal_photos.pop()) ])
           
    n_slides = len(horizontal_photos) + len(vertical_photos)//2

    # This runs for N^2 iterations...
    pbar = tqdm(range(n_slides))
    for i, _ in enumerate(pbar):

        total_photos = len(horizontal_photos)+len(vertical_photos)

        if total_photos == 0:
            break

        best_idx_h, score_h = find_greedy_match_photo(show.slides[-1], horizontal_photos)
        best_idx_v, score_v = find_greedy_match_photo(show.slides[-1], vertical_photos)

        if score_h > score_v:
            # Next slide should be horizontal
            show.add_slide(Slide(horizontal_photos.pop(best_idx_h)))
        else:
            # Next slide has vertical images
            best_photo = vertical_photos.pop(best_idx_v)

            # Find the next best vertical image to include
            best_idx, _ = find_greedy_match_vertical_photo(show.slides[-1], best_photo, vertical_photos)
            show.add_slide(Slide(best_photo, vertical_photos.pop(best_idx)))

        avg_score = show.score/len(show.slides)
        pbar.set_postfix(score=show.score, avg_score=avg_score)

    return show

if __name__ == "__main__":
    
    photos = parse_input(input_file)

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    if len(sys.argv) > 2:
        if sys.argv[2] == "similarity":
            similarity = generate_similarity_matrix(photos)
            np.save("./results/similarity_{}".format(base_name), similarity)

    # Get slides and generate a slideshow
    
    # Fast, scores about 1M
    #slides = get_vertical_slides_simple(photos) + get_horizontal_slides(photos)
    #show = greedy_slideshow_slides(slides)

    # A bit slower, scores 1.185M
    show = greedy_slideshow(photos)

    print("Slideshow score: ", show.score)

    os.makedirs("./results", exist_ok=True)
    show.save("./results/{}_{}_{}.txt".format(base_name, show.score, time.time()))
