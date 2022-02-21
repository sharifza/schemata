#!/usr/bin/env bash
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Downloading Faster-RCNN Checkpoint... ${NC}"
gdown --id 11zKRr2OF5oclFL47kjFYBOxScotQzArX --output vg-faster-rcnn.tar

echo -e "${GREEN}Downloading MotifNet Checkpoint... ${NC}"
gdown --id 12qziGKYjFD3LAnoy4zDT3bcg5QLC0qN6 --output vgrel-motifnet-sgcls.tar

echo -e "${GREEN}Done. ${NC}"