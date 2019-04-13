import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


Doc1 = 'Wimbledon is one of the four Grand Slam tennis tournaments, the others being the Australian Open, the French Open and the US Open.'
Doc2 = 'Since the Australian Open shifted to hardcourt in 1988, Wimbledon is the only major still played on grass'

doc_set = [Doc1, Doc2]

vectorizer = CountVectorizer(ngram_range=(2, 2))

term_count = vectorizer.transform(doc_set)

print(vectorizer.get_feature_names())

x=vectorizer.fit_transform(doc_set)

df = pd.DataFrame(x.toarray(), columns = vectorizer.get_feature_names())

print(df)
