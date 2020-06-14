# Open Syllabus Citation Annotator False Positive Filter

## Introduction
This project is focused on identifying false positives within the open syllabus citation matcher.  It uses mlflow to organize metrics and parameters so that one can compare different models and methods to see areas of gains.  The project is in a state in which one could easily come in and try different model architectures and quickly see if they perform better than any of the existing models.  It is also easy to replicate all the experiments conducted.

## Replicate experiments
Make sure you have mlflow installed:

    pip install mlflow

Clone the repo the cd into the working directory:

    cd ./opensyllabus-fp-filter 

To run all the experiments (including training) and steps in order simply:

    mlflow run . -P pth=./data/raw/matches.json

That will run the data cleaning, the baseline, and the experiements with naive bayes, once that has run start the mlflow ui with:

    mlflow ui

Now you can inspect each experiment and see which is performing best.  On my machine the model using just the lengths of the author, middle and title has an f1 score of 0.854 and can run inference on the test set in 0.003 seconds, so this seems to take the cake for the day.

## Performance
We are concerned with speed and reducing the amount of false positives of which we have many.  As such we want to record the amount of time inference takes as well as the F1 score.

## Baseline
It's looks like there is a some differentiation between the true and false labels on length of middle and length of title.
Our baseline will label the value False if title is less 17 or if the middle length is greater than 13 otherwise it's positive.

<!-- Add images and stats -->


### Results



## Models
### Processing
