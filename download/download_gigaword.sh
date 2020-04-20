perl download/gdown.pl https://drive.google.com/file/d/0B6N7tANPyVeBNmlSX19Ld2xDU1E/view summary.tar.gz

tar -zxvf summary.tar.gz

cd sumdata/train/

gzip -d train.title.txt.gz
gzip -d train.article.txt.gz
