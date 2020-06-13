# Open Syllabus Citation Annotator False Positive Filter

## Introduction
This project is focused on identifying false positives within the open syllabus citation matcher.  It uses mlflow to organize metrics and parameters so that one can compare different models and methods to see areas of gains.  The project is in a state in which one could easily come in and try different model architectures and quickly see if they perform better than any of the existing models.  It is also easy to replicate all the experiments conducted.

## Replicate experiments
Make sure you have mlflow installed
    pip install mlflow

To run all the experiments (including training) and steps in order simply:
    mlflow run . ./data/raw/matches.json

## Performance
We are concerned with speed and reducing the amount of false positives of which we have many.  As such we want to record the amount of time inference takes as well as the F1 score.

## Baseline
It's looks like there is a some differentiation between the true and false labels on length of middle and length of title.
Our baseline will label the value False if title is less 17 or if the middle length is greater than 13 otherwise it's positive.

<!-- Add images and stats -->


### Results



## Models
### Processing
