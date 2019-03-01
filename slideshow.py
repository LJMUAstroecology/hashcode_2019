from slide import Slide
from photo import Photo

class Slideshow:
    def __init__(self, slides = None):
        self.slides = []
        self.score = 0

        if slides is not None:
            self.add_slides(slides)
    
    def add_slides(self, slides):
        """
        Adds a list of slides to the slideshow
        """
        for slide in slides:
            self.add_slide(slide)
    
    def add_slide(self, next_slide):
        """
        Adds a slide the slideshow
        """
        if len(next_slide.photos) == 0:
            return

        if len(self.slides) == 0:
            # If the slideshow is empty, the previous slide is null
            prev_slide = Slide([])
        else:
            # Otherwise it's the last slide we added
            prev_slide = self.slides[-1]
        
        # The slideshow score is the sum of the transition scores

        self.score += prev_slide.score(next_slide)
        self.slides.append(next_slide)

    def save(self, filename):
        """
        Write competition output
        """
        with open(filename, "w") as of:
            of.write("{}\n".format(len(self.slides)))
            for slide in self.slides:
                ids = []
                for photo in slide.photos:
                    ids.append(photo.index)
                
                of.write(" ".join([str(id) for id in ids]))
                of.write("\n")