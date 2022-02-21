#!/usr/bin/env bash
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}(1/6) -- Downloading Visual-Genome dataset... ${NC}"
wget -c "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
wget -c "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"

echo -e "${GREEN}(2/6) -- Extracting Visual-Genome images... ${NC}"
mkdir -p visual_genome
unzip -q -d visual_genome/ images.zip
unzip -q -d visual_genome/ images2.zip

echo -e "${GREEN}(3/6) -- Merging the image folders... ${NC}"
rsync -a visual_genome/VG_100K_2/ visual_genome/VG_100K
mv visual_genome/VG_100K visual_genome/images
rm visual_genome/VG_100K_2 -r

echo -e "${GREEN}(4/6) -- Downloading Visual-Genome images meta-data... ${NC}"
wget -c "http://cvgl.stanford.edu/scene-graph/VG/image_data.json" -P stanford_filtered/

echo -e "${GREEN}(5/6) -- Downloading Visual-Genome scene graph relations... ${NC}"
wget -c "http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5"  -P stanford_filtered/

echo -e "${GREEN}(6/6) -- Downloading Visual-Genome scene graph information... ${NC}"
wget -c "http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json" -P stanford_filtered/

echo -e "${GREEN}Done. ${NC}"