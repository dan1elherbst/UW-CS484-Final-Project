wget -i data/kitti_archives.txt -P data/
unzip "data/*.zip" -d data/
rm data/*.zip
mv data/2011_09_26 data/train
mv data/2011_09_28 data/val