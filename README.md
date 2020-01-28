## Purpose:
    The idea for this code was to be able to predict the success chance of a kidney transplant and the length of time that the kidney would ast in the patient.
    
## Process:
    Through out the semester, my collegues and I would meet to discuss different methods to use in our respective Networks. While our networks are fairly simple, the biggest problem 
    we faced was feature selection as our dataset was around 130,000 patients each with about 1000 features. The initial dataset contained a large amount of null values, so we had 
    decided to replace those values with -1. After a while, we began to think that the -1 values were skewing the data, so we began to look for ways to get rid or replace the values.
    
## Output:
    For most of our research, we were using cross validation scores gathered from the Mean Ablsolute Error of our model's predictions when sent through our nets. We again began to 
    think that this was a skewed measurment of our data. So we just outputed the MAE given using stratified K-fold cross validation.
    