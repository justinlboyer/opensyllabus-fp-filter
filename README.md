# Open Syllabus Citation Annotator False Positive Filter

## Performance
We are concerned with speed and reducing the amount of false positives of which we have many.  As such we want to record the amount of time inference takes as well as the F1 score.

## Baseline
It's looks like there is a some differentiation between the true and false labels on length of middle and length of title.
Our baseline will label the value False if title is less 17 or if the middle length is greater than 13 otherwise it's positive.

<!-- Add images and stats -->


### Results



## Models
### Processing
