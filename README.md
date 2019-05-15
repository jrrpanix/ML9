# ML9
### Machine Learning NYU Spring 2019 
###### John Reynolds and Jeremy Lao

#### writeups
1. ./presentation
2. MLPaper.pdf - paper submitted for the class
3. MLPaper.txt - latex version

#### code
1. code in /src
2. model_analysis.py - work horse to compare models
3. to run stacked
4. /src/model_analysis.py --data minutes speeches minutes --stack --ngram 3,5 5,10, 10,15 -o ./myoutput
5. to run non-stacked 
6. /src/model_analysis.py --data minutes speeches minutes --ngram 3,5 5,10, 10,15 -o ./myoutput
7. modelutils.py : reads documents puts into pandas DataFrame: stacks the data : splits data : CountVectorizer
8. simulator.py : generates random text and simulates sentimate analysis
9. model2_pr.py : compares NB, LL, SVM,
10. textScraper.py : scrapes documents from federal reserve websites, gets speeches, statements, minutes
11. fedActionFromTargetRate.py : looks at historical federal erserve action
12. model2_lda.py for LatentDirichletAllocation
13. organizeDocuments.py : puts time stamps on files names and orgnizes docs into statemnts, speeches, minutes
12. neural nets ./src/model3_nn.py

#### output
1. output from analysis in csv files
2. output/naivebayes_20190513.csv
3. output/logisticlasso_20190513.csv

#### various analysis on output
1. python analysis.py -i ../analysis/all_lasso.csv ../analysis/naive.csv -r psize -l 1.1 -o SizeVF1
2. Sparsity vs F1
3. python analysis.py -i ../analysis/all_lasso.csv ../analysis/naive.csv -r psparse -l 0.0025 -o SparsityVF1

#### data for project
1. ./text/minutes    - meeting minutes
2. ./text/speeches   - fed speeches
3. ./text/statements  

#### data for simulator
1. ./simdata
2. 20k.txt 20 thousand common english words
3. neg_words.txt : note attribution in file
4. pos_words.txt : note attribution in file

#### some results from study
1. N-grams with CountVectorizer can generate millions of features and extremely sparse matricies
2. Logistic Lasso with slight penalty C=50.0 a=1/50.0 worked best
3. Naive Bayes does well in extreme sparsity
4. Stacking Documents reduces sparsity
5. N-grams between 4 and 20 words were the best
6. N-grams above 20 start creating extreme sparse matricies
7. Logistic Lasso with large regularization parameter does not do well with sparse matricies
8. TFIDF produced similiar results to CountVectorizer
9. TFIDF had slightly lower feature counts

