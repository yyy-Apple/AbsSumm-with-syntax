import os

if not os.path.exists('all.oracle'):
    os.system('perl download/gdown.pl https://drive.google.com/open?id=1q9X1kiDVn3JCFJUs8pcRe6OPIH-8V4jB all.oracle')
    os.system('bash download/download_gigaword.sh')

articles = open('sumdata/train/valid.article.filter.txt').readlines()
oracles = open('all.oracle').read().split('\n\n')[:-1]

assert len(articles) == len(oracles)

processed_articles, processed_oracles = [], []
for article, oracle in zip(articles, oracles):
    if '# #' in oracle or 'SEP' in oracle:
        continue
    processed_articles.append(article.strip())
    processed_oracles.append(oracle.strip())

print('\n'.join(processed_articles[:100000]), file=open('train.article', 'w'))
print('\n\n'.join(processed_oracles[:100000]), file=open('train.oracle', 'w'))

print('\n'.join(processed_articles[100000:110000]), file=open('dev.article', 'w'))
print('\n\n'.join(processed_oracles[100000:110000]), file=open('dev.oracle', 'w'))

print('\n'.join(processed_articles[110000:120000]), file=open('test.article', 'w'))
print('\n\n'.join(processed_oracles[110000:120000]), file=open('test.oracle', 'w'))
