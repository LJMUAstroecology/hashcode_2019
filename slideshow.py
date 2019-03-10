from slide import Slide
from photo import Photo

class Slideshow:
    """
    Helper class to store an image
    """

    def __init__(self, slides = None):
        """
        Instantiate a slideshow

        Input:

         - slides: optional, a list of slides to add to the slideshow
        """
        self.slides = []
        self.score = 0
        self.photos = {}

        if slides is not None:
            self.add_slides(slides)
    
    def add_slides(self, slides):
        """
        Adds a list of slides to the slideshow

        Input:

        - slides: list of slides
        """
        for slide in slides:
            self.add_slide(slide)
            """
            for photo in slide.photos:
                if photo in self.photos:
                    raise ValueError("Duplicate image")
                else:
                    self.photos[photo] = 1
            """
    
    def add_slide(self, slide):
        """
        Adds a slide the slideshow and updates the score

        Input:

        - slide: slide to be added
        """
        if len(slide.photos) == 0:
            return

        if len(self.slides) == 0:
            # If the slideshow is empty, the previous slide is null
            prev_slide = Slide()
        else:
            # Otherwise it's the last slide we added
            prev_slide = self.slides[-1]
        
        # The slideshow score is the sum of the transition scores

        score = prev_slide.score(slide)
        self.score += score
        self.slides.append(slide)

    def save(self, filename):
        """
        Write competition output

        Input:

        - filename: file to save the slideshow to
        """
        with open(filename, "w") as of:
            of.write("{}\n".format(len(self.slides)))
            for slide in self.slides:
                ids = []
                for photo in slide.photos:
                    ids.append(photo.index)
                
                of.write(" ".join([str(id) for id in ids]))
                of.write("\n")