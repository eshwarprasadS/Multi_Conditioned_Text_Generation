# Multi_Conditioned_Text_Generation
Conditional Text Generation using GPT-2 with multiple conditioning channels : Titles, Verdicts and Anchors. Final Project for Applied NLP at USC.

# Introduction
In this project, we attempt to improve conditioned
neural story generation. Specifically, given a title
and a verdict, the task of our neural model is to
generate a short, coherent first-person story that is
not only consistent with the title but also adheres
to the verdict. The verdict label chosen can either
be one that sheds a positive or negative light on the
narrator.

![alt text](aita_sample_annotated.png)

## Novel Contributions

- we propose a neural text generation system that
generates short compelling narratives conditioned
on a title,verdict pair. A verdict provides a diversifying
‘seed’ and supplements text generation with
additional context and direction.

- Additionally, we use an anchor-based
approach for gaining control and ensuring event
coherence in the generated text.

# Method

## Dataset

To power this corpus-driven application, we use
Reddit API(Reddit, 2019) to extract 121,634 posts
from r/AITA between January 1, 2014 and January
1, 2022.

In accordance with our data exploration, we perform
the following preprocessing steps:
- Retain only those posts which have more than
5 upvotes to make sure that posts and the corresponding
verdicts are meaningful.
- Retain only those posts with NTA and YTA
verdicts.
- Downsampling to balance the dataset between
the classes YTA and NTA (originally, YTA-
26%, NTA-74%).
- Retain posts with over 105 and under 596
tokens (5th - 95th percentile of word token
distribution in the dataset).

The resulting dataset consisting of 38,714 posts is
divided into two parts:
- Training Data - 37,751 posts
- Inference Data - 963 posts

Further, we use two methods to extract keyphrases
for each post:
- RAKE
- KeyBERT
