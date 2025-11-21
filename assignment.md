Assignment 4: Fine-tuning Language Models
Due date: Friday, November 15th, 11:59PM
This is an individual assignment. Please do not share your code or solutions with anyone else.
Please submit your work on Gradescope.
IMPORTANT: Please start early. Some parts of this assignment require GPU computation.
For this assignment, you may want to use the Courant GPUs described on the class website.
If you choose to do this, instead of sshing into access.cims.nyu.edu, please ssh into,
servers such as i7.cims.nyu.edu, or i8.cims.nyu.edu, or i9.cims.nyu.edu.
1. Part I: Data augmentation
In this part, we will explore data augmentation for sentiment analysis. We will use the IMDB dataset,
which contains 25,000 reviews labeled as positive or negative. The purpose of this assignment is to
get familiar with training and testing a model using data augmentation techniques.
To get started, clone the following repository:
git clone https://github.com/nyu-dl/NLP_hw4
cd NLP_hw4
Then, follow the instructions below.
1.1 Environment Setup
We recommend using Python 3.9 and pip to install dependencies. You may also use conda if you
prefer. First, create a virtual environment:
python3.9 -m venv hw4-part-1-nlp
source hw4-part-1-nlp/bin/activate
Then, install the dependencies.
pip install -r requirements.txt
You will also need to install nltk data files:
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
1.2 Part I, Question 1: Designing your transformation (10 points)
In part I, you will define a transformation of the dataset. Your transformation should take a
string as input, and output a string as output, and should modify the input string in a way that
could reasonably be encountered at test time.
For instance, synonymous word replacement is one possible transformation, or introducing small
typos. Please exercise good judgment: transformations should be “reasonable”.
You should describe your transformation in writing. In the PDF write-up, please clearly describe
exactly what your transformation does. This explanation should be sufficiently detailed such that
someone else could re-implement it from your description. You should also include why this is a
reasonable transformation, in a few sentences.
To check that your transformation behaves somewhat sensibly, you can run:
python3 main.py --train --eval
This will train a baseline model on the original IMDB dataset and evaluate using the original dataset.
In NLP_hw4 folder, you can find train and eval folders each containing original data for train
and dev sets.
Make sure that your code is run properly and that your transformation does not introduce issues.