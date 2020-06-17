# Open Syllabus Citation Annotator False Positive Filter

## Introduction
This project is focused on identifying false positives within the open syllabus citation matcher.  It uses mlflow to organize metrics and parameters so that one can compare different models and methods to see areas of gains.  The project is in a state in which one could easily come in and try different model architectures and quickly see if they perform better than any of the existing models.  It is also easy to replicate all the experiments conducted.

## Replicate experiments
Make sure you have mlflow installed:

    pip install mlflow

Clone the repo then cd into the working directory:

    cd ./opensyllabus-fp-filter 

To run all the experiments (including training) and steps in order simply:

    mlflow run . -P pth=./data/raw/matches.json

That will run the data cleaning, the baseline, and the experiements with naive bayes, once that has completed start the mlflow ui with:

    mlflow ui

Navigate to url (typically http:127.0.0.1:5000). Now you can inspect each experiment and see which is performing best.  On my machine the model using just the lengths of the author, middle and title has a F1 score of 0.854 and can run inference on the test set in 0.003 seconds, so this seems to take the cake for the day.

## Discussion of results
### 2020-06-15 Update
After working more with the data, it appears that the validation set is likely more difficult than the test set.  This is something that needs to be verified and the data may need to be reshuffled.

We are concerned with speed and reducing the amount of false positives of which we have many (billions in production).  As such, we want to record the amount of time inference takes as well as the F1 score.  I choose the F1 score to be my metric of choice, because I know that we have an abundance of false positives, so I need to have high recall, but I don't want to remove alot of actual books so I need to balance the precision as well.  Since the F1 score is the harmonic mean between precision and recall it fits the bill perfectly.  It might be worthwhile to come back at a later date and identify if we want to weight precision or recall more, but for the purpose of this exercise that is likely unnecessary.

### Baseline
I used the 'explore.py' file to identify a metric that could be used for a simple thresholded model.  As you can see from the images, we have some good differentiation between the classes on title and middle length.
![Middle Length vs. Class](/images/middle_len.png)

The differentiation  between the length of the strings between m1 and m2 is so prominent I opted to use that exclusively for the baseline.  Below I included the plot for the title lengths as well since there is definetly more fruit to harvest there, but for the sake of time decided not to optimize the thresholds between middle and title length.
![Title Length vs. Class](/images/title_len.png)

I set the middle length threshold to 12 because 75% of all False labels occur above that value (on the train set) and about half of all True values occur below.[^1]

[^1]: It should be noted that the plots are over the entire dataset, so there is a bit of data leakage there, but since this is just our baseline, shouldn't be too problematic.

As stated above I left the title threshold at 0, I'm confident that we can push the metrics of the baseline up by tuning it, but I just wanted a high water mark I should surpass.

The results of the said baseline on the test set are and f1 score of 0.702 and an inference time of 0.215.  Not too bad.


## Modeling
I choose to focus on the getting a good workflow established so that any model could be easily deployed or further itereation of models could be completed if desired.  Additionally I noticed that the baseline was doing very well, I wanted something that could essentially tune the baseline, but do well with small data, as well as be fast at prediction time.  Additional points for being able to extrapolate.  To this end I decided to start modeling with the naive bayes family.  XGboost is typically my go to, but because I knew I'd be working with character n-grams, I choose not to use XGboost because I thought it might struggle extrapolating on unseen counts.

### Gaussian Naive Bayes on Word Lengths
The first model was to just try and learn an "optimal baseline" essentially, to this end we have an approximately normal feature (the lengths), upon writing this I realized I'd like to come back and model it with a multinomial, because arguably that is more appropriate.  Anyhow this model is super fast running inference on the test set in about 0.003 seconds on my computer and the f1 score is the best out of the family of naive bayse at 0.854

### Multinomial on Character 2-gram Counts
Since the counts are ordinal and sparse I opted to use the multinomial formalization of naive bayes.  This had decent performance, still exceeding the baseline with a F1 score of 0.813 and in inference time of 0.032.  Not too bad.

### Gaussian on TIFIDF of 2-gram
Lastly I ran a model on the same counts as above but transformed by term frequency inverse document frequency.  This gives us continuous features again (albeit sparse), so I opted to use the gaussian formulation.  This model performed better than the multinomial model with an F1 score of 0.813, it cost a bit more clocking in with an inference time of 0.054 seconds.

### Deep Learning (Updated 2020-06-15)
#### Embedding + LSTM (baseline)
Since deep learning models can also get needlessly complex, I opted to start with a simple model, but make sure my pipeline was lock tight before I started iterating on the model.  The model I started with is a simple LSTM that ingests the characters of the middle string and uses the last unit in the sequence to make the prediction on the classification task.  This simple model is doing alright after 5 epochs it is hitting around 0.788 on F1 on the test set, and it is not too heavy clocking in at 0.372 seconds on the wall.  But this is not nearly good enough to run in production.

The models I intend to iterate on in order of priority (based on what is likely to lead to rapid gains):

1. ~~Ingest the author, middle, and title~~
2. Use Transformer architecture
3. Code up the smooth inverse frequency method and see if that works well with just embeddings
4. Train a semi-suprervised model to predict if the words (characters) are from an author, middle or title.  Then transfer this learning to another model for the task at hand.  Try 3 again with the new encoding.
4. Train a semi-supervised model with the above task as well as with masking of characters/words.  Try 3 again with the new encoding.

### Conclusion
Given how efficient the gaussian naive bayes on word lengths in conjunction with how well it performed this would be a great candidate for production.  If we went down that road I would want to conduct a secondary analysis to ensure that it is performing as expected, specifically I'd like to inspect many of the records and ensure that the predictions are agreeing with what I expect.  The beauty of the leg work in the set up I put together is all that would be needed to deploy it to say sagemaker is to log the model and deploy it to sagemaker by calling (roughly)[^2]:
[^2]: One would still need to set up aws appropriately and blah, blah, blah, but it is much easier than other ways.

    mlflow.sklearn.log_model(gnb, artifact_path)
    mlflow.sagemaker.deploy(app_name, model_uri)


Lastly, I was really hoping to train a character embedding then use smooth inverse frequncy method.  I think this could blow most things out of the water and be very fast to boot.  Other ideas are obviously logistic regression, which would not be too difficult to throw in (unlikely to see much gain though as well), pretty much copy the naive bayes script and replace the naive bayes models with logistic regression (another benefit of the structure I created).  If I have time tomorrow, I plan on circling back to the above.  Another avenue that occured to me that might be good is fasttext.  


## Appendix
Overview of the files:

`MLproject`: this file sets up all the scripts so that they could be invoked using the mlflow cli individually

`explore.py`: my exploratory data analysis

`utils.py`: some utitlites I wrote that I used across scripts

`main.py`: the script that invoked all the steps and runs data preprocessing, baseline, and model training and testing

`conda.yaml`: the environment, this file is used in the MLproject.

`clean.py`: this script converts the file to csv, adds in some simple features and generated the count vectorizer which is serialized into a model folder

`baseline.py`: this script holds the baseline model and invokes it on the test set, logging metrics to mlflow

`naive_bayes.py`: this script holds the various naive bayes models and logs of the metrics to mlflow