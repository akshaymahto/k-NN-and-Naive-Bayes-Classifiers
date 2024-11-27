

# Assignment 2: k-Nearest Neighbors and Na ̈ıve Bayes Classifiers
10 points
September 27, 2024
Abstract
ATTENTION: this assignment should be completed individually. And I will use
tools to check your codes against your peers’ submissions! In this assignment, you
are going to use KNNs and Na ̈ıve Bayes for solving real-world questions. For each
problem, the questions are designed to modify the data sequentially.
# Problem 1: k-Nearest Neighbors for marketing campaign (6
points)
Download the dataset bank-full.csv from Canvas. For this assignment, we will be using the bank
marketing dataset from UCI. The data has 17 attributes and is related to marketing campaigns (phone
calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe
a term deposit (the variable “y” in the dataset).
Data Exploration
Open the file bank-names.txt from Canvas and carefully read the attribute information to understand
what information is stored in each attribute, what values each attribute can take and so on.
# 1. (0.1pt) Download the dataset and store it in a dataframe in R. Note: the attributes are
separated by semicolon, make sure you set “sep” option correctly inside the function read.csv.
In addition, some variables use “unknown” for missing values. Convert all “unknown” values to
NA. You can do so by setting “na.strings” parameter to “unknown” when you read the file.
# 2. (0.2pts) Specify the type of each variable as follows:
– Specify whether the variable is categorical(qualitative) or numeric(quantitative)?
– For categorical variables, specify whether it is a nominal variable or an ordinal variable.
– For numerical variables, specify whether it is an interval-scaled variables or a ratio-scaled
variables?
# 3. (0.2pts) Get the frequency table of the target variable “y” to determine how many observations
(examples) are present in each category (y = yes and y = no). Is the target variable “y” balanced?
In other words, do both categories (y = yes and y = no) have roughly the same number of
observations (examples)?
# 4. (0.5pts) Explore the data in order to investigate the association between the target variable
y and other variables in the dataset. Which of the other variables are associated with y? Use
appropriate plots and statistic tests to answer this question. Check the table at the end of this
assignment to see which statistic test and plot you should use.
Based on your data exploration above, keep the variables you have found in q4, which
have association with the target variable y, and remove the other variables.
1
Data Preparation
# 5. (0.1pt) Use the command colSums(is.na(<your dataframe>)) to get the number of miss-
ing values in each column of your dataframe. Which columns have missing values?
# 6. (0.4pts) There are several ways we can deal with missing values. The easiest approach is to
remove all the rows with missing values. However, if a large number of rows have missing values,
then removing them will result in loss of information and may affect the classifier performance. If
a large number of rows have missing values, then it is typically better to substitute missing values.
This is called data imputation. Several methods for missing data imputation exist. The most
na ̈ıve method (which we will use here) is to replace the missing values with mean of the column
(for a numerical column) or mode/majority value of the column (for a categorical column). We
will use a more advanced data imputation method in a later module.
For now, replace the missing values in a numerical column with the mean of the column and the
missing values in a categorical column with the mode/majority of the column. After imputation,
use colSums(is.na(<your dataframe>)) to make sure that your dataframe no longer has
missing values.
# 7. (0.2pts) Set the seed of the random number generator to a fixed integer, say 1, so that I can
reproduce your work: set.seed(1). Then, randomize the order of the rows in the dataset.
# 8. (0.3pts) This dataset has several categorical variables. One way to deal with categorical
variables is to assign numeric indices to each level. However, this imposes an artificial ordering
on an unordered categorical variable. For example, suppose that we have a categorical variable
primary color with three levels: “red”, “blue”, “green”. If we convert “red” to 0 , “blue” to 1,
and “green” to 2, then we are telling our model that red < blue < green which is not correct.
A better way to encode an unordered categorical variable is to do one-hot encoding. In one-hot
encoding we create a dummy binary variable for each level of a categorical variable. For example,
we can represent the primary color variable by three binary dummy variables, one for each color
(red, blue, and green). If the color is red, then the variable red takes value 1 while blue and
green both take the value 0.
Do one-hot encoding for all your unordered categorical variables (except the target variable y).
You can use the function one hot from “mltools” package. Please refer to this link for usage.
Use option “DropUnusedLevels=True” to avoid creating a binary variable for unused levels of a
factor variable.
Please note that the one hot function takes a “data table” not a “dataframe”. You can convert
a “dataframe” to “datatable” by using as.data.table method, see this link. Make sure to use
“library(data.table)” before using as.data.table method.
You can covert a “data table” back to a “dataframe” by using as.data.frame method, see this
link.
Training and Evaluation of ML models
# 9. (0.1pts) Split the data into training and test sets. Use the first 36168 rows for training and
the rest for testing.
# 10. (0.2pts) Scale all numeric features using z-score normalization. Note: Don’t normalize your
one-hot encoded variables.
# 11. (1pt) Use 5-fold cross validation with KNN on the training set to predict the “y” variable
and report the cross-validation accuracy. (Please use crossValidationError function in week
# demo codes and modify it to compute accuracy instead of error, where the accuracy is simply
equal to 1 - error).
# 12. (1pt) Tune K (the number of nearest neighbors) by trying out different values, i.e., K =
1, 5, 10, 20, 50, 100, 200. Draw a plot of cross validation accuracy for different values of K.
Which value of K seems to perform the best on the cross validation? (Note: the higher the cross
validation accuracy (or the lower the cross validation error), the better the model is. You can
2
find a similar example in week 4 demo codes). This question might take several minutes to run
on your machine, sit tight.
# 13. (0.5pts) With the best value of K you found above, use knn function to get the predicted
values for the target variable y in the test set.
# 14. (0.5pts) Compare the predicted target (y) with the ground truth target (y) in the test set
using a cross table.
# 15. (0.3pts) Based on the cross table above, what is the False Positive Number and False
Negative Number of the knn classifier on the test data? False Positive (FP): The number of all
true negative (y = ”no”) observations that the model incorrectly predicted to be positive (y =
”yes”). False Negative (FN): The number of all true positive (y = ”yes”) observations that the
model incorrectly predicted to be negative (y = ”no”).
# 16. (0.2pts) Consider a majority classifier which predicts y=”no” for all observations in the test
set. Without writing any code, explain what would be the accuracy of this majority classifier?
Does KNN do better than this majority classifier? .
# 17. (0.2pts) Explain what is the False Positive Number and False Negative Number of the
majority classifier on the test set and how does it compare to the False Positive Number and
False Negative Number of the knn model you computed in question 15.
# Problem 2: Applying Na ̈ıve Bayes classifier to sentiment clas-
sification of COVID tweets (4 points)
For this problem you are going to use Corona NLP train.csv dataset, a collection of tweets pulled
from Twitter and manually labeled as being “extremely positive”, “positive”, “neutral”, “negative”,
and “extremely negative”.
The dataset is from this Kaggle project. You can download the dataset from Canvas.
# 1. (0.1pt) Read the data and store in the dataframe. Take a look at the structure of data
and its variables. We will be working with only two variables: OriginalTweet and Sentiment.
Original tweet is a text and Sentiment is a categorical variable with five levels: “extremely
positive”, “positive”, “neutral”, “negative”, and “extremely negative”. Note: The original tweet
variable has some accented character strings. Set fileEncoding=”latin1” parameter inside the
read.csv method to ensure those characters are read correctly.
# 2. (0.1pts) Set the seed of the random number generator to a fixed integer, say 1, so that I can
reproduce your work: set.seed(1). Then, randomize the order of the rows in the dataset.
# 3. (0.2pts)Convert sentiment into a factor variable with three levels: “positive”, “neutral”,
and “negative”. You can do this by labeling all “positive” and “extremely positive” tweets as
“positive” and all “negative” and “extremely negative” tweets as “negative”. Then, take the
summary of sentiment to see how many observations/tweets you have for each label.
# 4. (0.5pts) Create a text corpus from OriginalTweet variable. Then clean the corpus, i.e., convert
all tweets to lowercase, perform stemming, remove stop words, remove punctuation, and a remove
additional white spaces.
# 5. (0.5pts) Create separate wordclouds for “positive” and “negative” tweets (set “max.words=100”
to only show the 100 most frequent words). Is there any visible difference between the frequent
words in “positive” vs. “negative” tweets?
# 6. (0.5pts) Create a document-term matrix from the cleaned corpus. Then split the data into
train and test sets. Use the first 32925 rows (roughly 80% of samples) for training and the rest
for testing.
3
# 7. (0.5pts) Remove the words that appear less than 50 times in the training data (Note: use
findFreqTerms in week 5 demo codes). And convert frequencies in the document-term matrix
to binary yes/no features (one-hot encoding).
# 8. (0.8pts) Train a Na ̈ıve Bayes classifier on the training data and evaluate its performance on
the test data. Plot a cross table between the model’s predictions on the test data and the true
test labels.
# 9. (0.4pts) Based on the cross table you plot above, what is the overall accuracy of the model?
(the percentage of correct predictions).
#  10. (0.4pts) Based on the cross table you plot above, what is the precision and recall of the
model in each category (negative, positive, neutral)?
Precision and Recall are two popular metrics for measuring the performance of a classifier on
each class and they are computed as follows:
P recision = T P/(T P + F P ); Recall = T P/(T P + F N ),
where T P is True Positive, F P is False Positive, and F N is False Negative.
For example, for the “neutral” class, T P will be the total number of neutral observations in
the test data that na ̈ıve bayes correctly predicted. F P will be the total number of none-neutral
observations in the test data that na ̈ıve bayes incorrectly predicted to be neutral. And F N will
be the total number of neutral observations in the test data that na ̈ıve bayes incorrectly predicted
to be none-neutral.
Precision for the neutral class answers this question: what percentage of observations that the
model classified as neutral are truly neutral? Recall for neutral answers this question: what
percentage of truly neutral observations was the na ̈ıve bayes model able to predict correctly as
neutral?
We will talk more about different evaluation metrics around module 12.
Numeric Ordinal Nominal
Numeric
Pearson;
Scatter Plots
Kendall or Spearman;
Scatter Plots
If nominal variable has
two groups:
- Two sample t-test
If nominal variable has
more than two groups:
- ANOVA
Side by side boxplots
Ordinal
Kendall or Spearman;
Scatter Plots
Kendall or Spearman;
Scatter Plots
Kruskal-Wallis test;
Side by side boxplots
Nominal
If nominal variable has
two groups:
- Two sample t-test
If nominal variable has
more than two groups:
- ANOVA
Side by side boxplots
Kruskal-Wallis test;
Side by side boxplots
Chi-Square test;
Mosaic Plots
Table 1: Bivariate analysis: what of statistic test and plot should you use?
4
What to turn in
You need to create an R notebook consisting of your answers/analysis to the questions outlined
above for problems one and two together with your R code you used to answer each question.
R Notebooks are valuable tools that combine the power of R programming with the flexibility of
dynamic, interactive documents. Every block in a notebook can be a text (Markdown) or an R code.
When you run the notebook, it will run all the code blocks and shows the narrative, the code blocks
and the output of each code block. Using R notebook is very easy and intuitive. You can watch these
two short Youtube tutorials to get started with R notebook:
• What is R notebook?
• How to use R notebook?
The submission must be in two formats
• A html file; You run all the code cells, get all the intermediate results and formalize your
answers/analysis, then you click ”preview” and this will create a html file in the same directory as
your notebook. You must submit this html file or your submission will not be graded.
Please note that if you knit your R notebook once, then you will lose the ”preview”
button! So do not knit!
• An Rmd file which contains your R notebook.
Please do not hesitate to post your questions on Ed discussion (this will give you
bonus points), or email me if you have any question.
