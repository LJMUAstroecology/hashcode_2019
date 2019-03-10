# Solution for Google Hash Code 2019

This is a simply greedy solution which scores around 1.185-1.19M points (with some randomness). Although it doesn't beat the 1.2M mark, this would have placed around 11th on the competition scoreboard (not the extended, where it's currently about 60th).

Hopefully this will be useful to someone!

The algorithm is pretty straightforward:

1. Pick a photo
2. Pick the best photo to match with
3. If horizontal, add to slideshow
4. If vertical, find the best other vertical photo to pair with
5. Repeat

## Running:

`python main.py <dataset.txt>`

You can also precompute photo-photo scores with:

`python main.py <dataset.txt> similarity`

This is not actually used in the algorithm, but you might find it interesting to explore the data.
