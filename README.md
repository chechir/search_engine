# Search Engine

This is a small experiment that performs a search in a Pandas dataframe. The idea is that the results are returned in a reasonable time and also that they are relevant. The data was obtained from here: https://www.kaggle.com/therohk/million-headlines

## The Solution

###Summary
The program creates a TF-IDF representation for every row in the data frame,, scoring every word (or n-gram) in the text. This produces a sparse matrix with around 1 million rows and n columns (n = number of n-grams obtained in the entire df). Then the program produces the equivalent TF-IDF matrix for a given query and calculates the cosine similarity between both matrices. This cosine similarity is used as the relevance score for each row in our df. Finally, the program returns the top 10 most relevant news headlines in the df for the query.

### Implementation
The implemented code is basically a class called 'SearchEngine'. This class performs mainly two tasks: 
1. Prepare the data set and build a 'model' based on the entire df (method: fit). 
2. Given a query, return the most relevant products in the df that match this query (method: get_results)

### Method *fit*
This method will perform the following actions: 

1. Prepare the data: It concatenates the 'name' and 'brand' columns into one column, converts the text to lower case for this new column, removes stop words and optionally stems the words using the Porter stemer.

2. Calculate a TF-IDF matrix: It will calculate the TF-IDF matrix for the entire catalog, using different n-grams. By default it uses 1 to 3 n-grams. More information on TF-IDF (Term frequency – Inverse document frequency) can be found in wikipedia: https://en.wikipedia.org/wiki/Tf%E2%80%93idf. More information about n-grams could be found in here: https://en.wikipedia.org/wiki/N-gram
For this task the program uses the sklearn package.

3. Save the objects: It saves the sparse matrix for the catalog and the sklearn models as attributes of the class


### Method *get_results*
This method will receive as a parameter a single query and it will perform the search in the catalog returning up to 10 results ordered by the ranking score. The ranking score will be obtained in the following way:

1. Preprocess the query using the same transformations used in the 'fit' method for the entire df (convert the text to lower case, remove stop words, stem words if it applies)

2. Obtain a TF-IDF representation of our query. This will be a sparse matrix of only 1 row

3. Calculate the cosine similarity between the matrix representation of our query and our entire catalog matrix. This vector will be our ranking score

4. Sort the catalog by the ranking score obtained in the previous step (in descending order) and return the top 10 most relevant products for our queries 


## Program performance

As the reader can guess, the hard work happens in the fit method, because it has to calculate the entire matrix. fortunately, the CountVecotrizer and TfidfTransformer classes use sparse matrices that are very efficient. 

### Timing
The fitting process takes around 6 seconds (without steming the words) and 20 seconds (steming the words). Those times were got using a laptop with CPU: Intel core i5, and RAM: 8 GB. 

Obtaining the results for a single query takes about 31ms seconds, which is under the benchmark of 50 ms, given by the problem constraints.


## Running the program

The program runs in Python 3.6 and uses the following external libraries: pandas, numpy, nltk, sklearn and stemming. The tests were performed using py.test

The requirements can be installed using the requirements.txt file: 
> search_engine\>pip install -r requirements.txt

In order to run the program, you need to copy the provided folder ("search_engine") into any folder defined as PYTHONPATH in your environment, and run the script called 'search.py' passing the path to your queries path as a parameter. Example:

> search_engine\>python search.py input/queries_example.txt

Make sure that your path to the data file (e.g. "abcnews-date-text.csv") is configured correctly in the config.py file. The data file is already provided in the "input" folder and the default path already points to it as a relative path.


## Tests

A set of 5 tests are provided in the tests 'folder'. They test the basic functionality of individual functions or methods. They also help to understand what the individual functions are returning. In addition, they provide a safety net for future deveopment.


## Comments

### Timing functions

A set of functions are provided in the file measure_time.py. The only purpose of this functions is to time the execution of the 'fit' and 'get_results' methods


## Execution examples:


Time Func: `fit` call took 106.5621s.

---------
results for "I dont like cricket, I love it"
0.5408148666722554, 20111227, lewis i dont like test cricket; i love it
0.4746528486047685, 20090312, i dont like it like that
0.3340697208835637, 20120329, they dont like the way i fight
0.30694877160059447, 20071226, its just not cricket
0.2991960409551563, 20170215, if you dont like it dont stand pauline hanson
0.28294334319152525, 20120103, for the love of cricket
0.2755673628490005, 20120726, farmers dont like killing wombats
0.275361313982031, 20090512, water dont like buybacks too bad
0.2667672646509306, 20060920, argentines dont like hewitt nalbandian
0.24620939591758922, 20140508, dunlop abbott unveiled and we dont like what we s
ee

Time Func: `get_results` call took 0.7550s.

---------
results for "global warming"
0.5613951968866028, 20110602, phelps global warming
0.5499402388144423, 20120329, fertilisers and global warming
0.5418577547620187, 20140213, measuring global warming
0.49252038363751555, 20060126, g g calls for action on global warming
0.47241626541682524, 20060705, govt under pressure on global warming
0.4699100778077073, 20070301, global warming threat very real
0.462902462587552, 20061215, report confirms global warming
0.45224195162868264, 20120427, global warming may have been under estimated
0.45139335853193596, 20060914, sun not to blame for global warming
0.44792689439511735, 20071111, tasmanians rally against global warming

Time Func: `get_results` call took 0.7390s.

---------
results for "how can I win kaggle competitions from my cell phone"
0.3073252418384191, 20030507, marijuana cell phone cover causes buzz
0.19290441501534525, 20030307, rowing competitions to go ahead
0.16061952072095026, 20030312, nsw competitions to crack down on umpire abuse
0.15987209211606965, 20160307, no new netball facilities for darwin competitions
0.15975332480223822, 20080804, football finals loom in canberra competitions
0.1593404420354847, 20110815, nsw dominates ekka cattle competitions
0.15758272221215727, 20130503, unique competitions at pooncarie field day
0.157152088597471, 20120502, phone scam
0.157152088597471, 20150205, phone scam
0.14979414894702056, 20131108, southern phone

Time Func: `get_results` call took 0.7610s.

---------
results for "what is the meaning of life"
0.45780324694283636, 20160114, scott how i started to bring meaning to my life
0.3445841844307364, 20150413, is there life out there
0.21090048747515452, 20160628, redefining the meaning of classical
0.20422949207406327, 20161031, the meaning of oscietras silks
0.18879347128493693, 20120910, students find meaning through murals
0.18451782539802117, 20161221, how to reclaim the true meaning of christmas
0.18055186982388416, 20120205, new life
0.1759452342157714, 20051019, road rage takes on new meaning
0.17388270868410746, 20170608, cathy freeman on finding meaning and success in l
ife after sport
0.16672632766008444, 20050425, anzac day now has wider meaning governor

Time Func: `get_results` call took 0.7530s.

---------
results for "donald trump riding an skate board"
0.3468294613562698, 20161114, the case for and against donald trump
0.34258589483762597, 20170204, donald trump art
0.335270219310989, 20170210, donald trump does have some opportunities to
0.33429143795499455, 20161109, what has donald trump promised to do
0.3335172910123638, 20161109, donald trump in a minute
0.3305372021868078, 20161004, donald trump on the apprentice
0.3124981391314799, 20170407, has donald trump been moonlighting
0.30864488134482976, 20171101, donald trump is at his zenith
0.27998088592533354, 20150806, donald trump republican debate
0.2722868141397767, 20170522, donald trump arrives in israel

Time Func: `get_results` call took 0.7440s.

---------
results for "some people like weird things, like pizza with pineapple"
0.37288499272114145, 20080822, why do people believe weird things
0.2646283517269077, 20090312, i dont like it like that
0.22009897027752606, 20091013, i hope you like it
0.21670226591504838, 20110215, with friends like these
0.21670226591504838, 20070213, with friends like these
0.2127994137659573, 20091224, why i like politicians
0.20998804675000202, 20120314, how do you like them apples
0.20698199066213852, 20090705, in like flynn
0.20698199066213852, 20080530, in like flynn
0.20504667428620807, 20170921, documentary its people like us shows phone use wh
ile driving

Time Func: `get_results` call took 0.7410s.


---------------

Author: Matias Thayer

email: chechir@gmail.com
