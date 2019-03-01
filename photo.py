class Photo:
    """
    Helper class to define an image
    """
    def __init__(self, id, tags, orientation = "H"):
        self.tags = set(tags)
        self.index = id
        self.orientation = orientation

def transition_score(tags_a, tags_b):
    score_1 = len(tags_a.intersection(tags_b))
    score_2 = len(tags_a.difference(tags_b))
    score_3 = len(tags_b.difference(tags_a))

    return min(score_1, score_2, score_3)