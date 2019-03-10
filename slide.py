from photo import Photo, transition_score


class Slide:
    """
    Helper class to define a slide
    """
    def __init__(self, photo=None, photo_2=None):
        """
        Instantiate a slide

        Input:

        - photo: a photo, can be horizontal or vertical
        - photo_2: a second photo, both photos must be vertical if used
        """
        self.tags = set()
        self.photos = []

        if photo is not None:
            self.add_photo(photo)

        if photo_2 is not None:
            assert(photo.orientation == "V" and photo_2.orientation == "V")
            self.add_photo(photo_2)

    def add_photo(self, photo):
        """
        Add a photo to the slide

        Input:

        - photo: photo to be added
        """
        self.photos.append(photo)
        self.tags.update(photo.tags)
    
    def remove_photo(self):
        """
        Remove a photo from the slide

        Input:

        - photo: photo to be removed
        """
        self.photos.pop()
        self.tags = self.photos[0].tags

    def score(self, next_slide):
        """
        Returns the "interest" score between this slide 
        and the provided slide.

        Input:

        - next_slide: slide to compare with
        """
        
        return transition_score(self.tags, next_slide.tags)