#!/bin/bash

for STYLE in filt4 filt3 fbot-2 "dual_3_-2" dbp4 win0 dbp3 inp2 dbp5 bot0 inp3
do


# Adversarial dissection
echo "Adversarially Testing ${STYLE}"
python src/test_adv_resnet.py \
    --expdir experiment/${STYLE}_resnet

echo "Adversarial dissection of ${STYLE}"
python -m netdissect \
   --model "src.customnet.CustomResNet(18, num_classes=100, halfsize=True)" \
   --pthfile experiment/${STYLE}_resnet/best_miniplaces.pth.tar \
   --perturbation perturbation/VGG-19.npy \
   --outdir experiment/${STYLE}_resnet/adv_dissect \
   --layers layer1 layer2 layer3 layer4 \
   --imgsize 128 \
   --batch_size 32 \
   --meta experiment/${STYLE}_resnet/adversarial_test.json \
   --size 10000 \
   --netname "Regularized ${STYLE} Resnet-18 adversarially tested"

echo "Dissecting ${STYLE}"
python -m netdissect \
   --model "src.customnet.CustomResNet(18, num_classes=100, halfsize=True)" \
   --pthfile experiment/${STYLE}_resnet/best_miniplaces.pth.tar \
   --outdir experiment/${STYLE}_resnet/dissect \
   --layers layer1 layer2 layer3 layer4 \
   --imgsize 128 \
   --batch_size 32 \
   --size 10000 \
   --netname "Regularized ${STYLE} Resnet-18 adversarially tested"

done

# Adversarial dissection
echo "Adversarial test for ${STYLE}"
python src/test_adv_resnet.py \
    --expdir experiment/resnet

python -m netdissect \
   --model "src.customnet.CustomResNet(18, num_classes=100, halfsize=True)" \
   --pthfile experiment/resnet/best_miniplaces.pth.tar \
   --perturbation perturbation/VGG-19.npy \
   --outdir experiment/resnet/adv_dissect \
   --layers layer1 layer2 layer3 layer4 \
   --imgsize 128 \
   --batch_size 32 \
   --meta experiment/resnet/adversarial_test.json \
   --size 10000 \
   --netname "Basline Resnet-18 adversarially tested"

# Adversarial dissection
echo "Testing ${STYLE}"
python src/test_adv_resnet.py \
    --expdir experiment/resnet

python -m netdissect \
   --model "src.customnet.CustomResNet(18, num_classes=100, halfsize=True)" \
   --pthfile experiment/resnet/best_miniplaces.pth.tar \
   --outdir experiment/resnet/dissect \
   --layers layer1 layer2 layer3 layer4 \
   --imgsize 128 \
   --batch_size 32 \
   --size 10000 \
   --netname "Basline Resnet-18 adversarially tested"
