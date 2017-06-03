# Proposal


## What is the problem that you will be investigating? Why is it interesting?

I'll be investigating the problem of detecting the genre of one song using convolutional neural networks and maybe using recurrent networks if I see
a way how this could be useful. I find this problem intersting first of all because I love music. This problem also could be applied in music understanding and
music recommendation.

If I have time I would like to explore how this could be accomplished using GANs and not only CNNs.

## What data will you use? If you are collecting new datasets, how do you plan to collect them?

I plan to use the Million Song Dataset. Probably a subset of it.

## What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations?

I'll be experimenting with different neural network architectures to see which one performs better.
There's an existing implementation described in http://benanne.github.io/2014/08/05/spotify-cnns.html. I plan to start testing that one and see how I can improve upon it after
analzing the information and the features it's extracting.

## What reading will you examine to provide context and background?

I'll look for papers that have explored this problem already.

Some ideas:
- http://benanne.github.io/2014/08/05/spotify-cnns.html
- https://courses.engr.illinois.edu/ece544na/fa2014/Tao_Feng.pdf
- https://arxiv.org/pdf/1607.02444.pdf
- http://papers.nips.cc/paper/5004-deep-content-based-%20music-recommendation.pdf


## How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?

I'll train with a subset of the MSD, validate with another subset of it and test with another subset of it.
