class Photo:
    """
    Helper class to store an image
    """
    def __init__(self, id, tags, orientation = "H"):
        """
        Instantiate an image

        Input:

         - id: index
         - tags: a list or set of tags
         - orientation: image orientation, "H" or "V"
        """
        self.tags = set(tags)
        self.index = id
        self.orientation = orientation

def transition_score(tags_a, tags_b):
    """
    Compute the transition/interest score between two sets of tags
    """
    score_1 = len(tags_a.intersection(tags_b))
    score_2 = len(tags_a) - score_1
    score_3 = len(tags_b) - score_1

    return min(score_1, score_2, score_3)