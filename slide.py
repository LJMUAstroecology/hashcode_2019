from photo import Photo, transition_score


class Slide:
    """
    Helper class to define a slide
    """
    def __init__(self, photos):
        self.tags = set()
        self.photos = photos

        # A slide has the union of the photos it contains
        for p in self.photos:
            self.tags.update(p.tags)

    def score(self, next_slide):
        """
        Calculates the "interest" score between this slide 
        and the one that follows.
        """
        
        return transition_score(self.tags, next_slide.tags)