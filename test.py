import photo
import slide
import pytest

def test_same_tags():
    p1= photo.Photo(id=0, tags=["cat", "house", "garden"])
    p2= photo.Photo(id=0, tags=["cat", "house", "garden"])
    s1 = slide.Slide(p1)
    s2 = slide.Slide(p2)

    assert(s1.score(s2) == 0)
    assert(s2.score(s1) == 0)


def test_different_tags():
    p1= photo.Photo(id=0, tags=["cat", "house", "garden"])
    p2= photo.Photo(id=0, tags=["sun", "beach", "sky"])
    s1 = slide.Slide(p1)
    s2 = slide.Slide(p2)

    assert(s1.score(s2) == 0)
    assert(s2.score(s1) == 0)

def test_fewer_tags():
    p1= photo.Photo(id=0, tags=["cat", "house", "garden"])
    p2= photo.Photo(id=0, tags=["cat", "house"])
    s1 = slide.Slide(p1)
    s2 = slide.Slide(p2)

    # Expect zero as b has no tags other than those in a

    assert(s1.score(s2) == 0)
    assert(s2.score(s1) == 0)

def test_equal_different_tags():
    p1= photo.Photo(id=0, tags=["cat", "house", "garden"])
    p2= photo.Photo(id=0, tags=["cat", "house", "sky"])
    s1 = slide.Slide(p1)
    s2 = slide.Slide(p2)

    assert(s1.score(s2) == 1)
    assert(s2.score(s1) == 1)

def test_equal_more_tags():
    p1= photo.Photo(id=0, tags=["cat", "house", "garden", "bear"])
    p2= photo.Photo(id=0, tags=["cat", "house", "sky", "dog"])
    s1 = slide.Slide(p1)
    s2 = slide.Slide(p2)

    assert(s1.score(s2) == 2)
    assert(s2.score(s1) == 2)

def test_equal_more_tags_2():
    p1= photo.Photo(id=0, tags=["cat", "house", "garden", "bear"])
    p2= photo.Photo(id=0, tags=["cat", "house", "sky", "dog", "cow"])
    s1 = slide.Slide(p1)
    s2 = slide.Slide(p2)

    assert(s1.score(s2) == 2)
    assert(s2.score(s1) == 2)
