mkdir -p maganatagtune
cd magnatagatune
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/clip_info_final.csv
# https://gist.github.com/4np/2913012
cat *.zip > combined.zip;zip -FF combined.zip --out combined-fixed.zip;rm combined.zip;yes A|unzip -qq combined-fixed.zip;rm combined-fixed.zip
rm mp3.zip*
cd ..
