#!/bin/bash

# Adversarial dissection of smoothed network
# echo "Adversarial test for ${STYLE}"
# python src/test_adv_resnet.py \
#    --expdir experiment/resnet

python -m netdissect \
   --model "src.smoothnet.SmoothedResNet18()" \
   --pthfile experiment/smoothnet/best_miniplaces.pth.tar \
   --outdir experiment/smoothnet/dissect \
   --layers layer1 layer2 layer3 layer4 \
   --imgsize 128 \
   --batch_size 32 \
   --size 10000 \
   --netname "Smoothed Resnet-18"

# Adversarial dissection
echo "Testing ${STYLE}"
python src/test_smoothnet.py \
    --expdir experiment/smoothnet

python -m netdissect \
   --model "src.smoothnet.SmoothedResNet18()" \
   --pthfile experiment/smoothnet/best_miniplaces.pth.tar \
   --perturbation perturbation/VGG-19.npy \
   --outdir experiment/smoothnet/adv_dissect \
   --layers layer1 layer2 layer3 layer4 \
   --imgsize 128 \
   --batch_size 32 \
   --meta experiment/smoothnet/smoothed_adv_test.json \
   --size 10000 \
   --netname "Smoothed Resnet-18 adversarially tested"
