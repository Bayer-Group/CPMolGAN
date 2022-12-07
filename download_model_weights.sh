cd cpmolgan
wget --load-cookies /tmpcookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KgVTglQ_LbKzzEkhGzEdZxBA9RBTXIFl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KgVTglQ_LbKzzEkhGzEdZxBA9RBTXIFl" -O model_weights.zip && rm -rf /tmp/cookies.txt
unzip model_weights.zip
rm model_weights.zip