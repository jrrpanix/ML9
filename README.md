# ML9
### Machine Learning NYU Spring 2019 
###### John Reynolds and Jeremy Lao

#### writeup
1.  ./presentation/MLPaper.pdf
2. latex version MLPaper.txt

#### code
1. code in /src
2. ./src/model_analysis.py
3. to run stacked
4. /src/model_analysis.py --data minutes speeches minutes --stack --ngram 3,5 5,10, 10,15 -o ./myoutput
5. to run non-stacked 
6. /src/model_analysis.py --data minutes speeches minutes --ngram 3,5 5,10, 10,15 -o ./myoutput
7. modelutils.py
8. reads documents puts into pandas DataFrame
9. stacks the data
10. split between test and train
11. clean.py : document cleaning
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


Observations from testing: 
  1.    Ngrams (bigrams and trigrams) increase the accuracy 
  2.    Logistic lasso performs the best
  3.    TFIDF vectorizer lowers the accuracy
