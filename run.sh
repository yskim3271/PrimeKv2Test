# chmod +x ./run.sh
# nohup ./run.sh > output.log 2>&1 &

echo "####################################"
echo "메모: PrimeK-Net 재현성 테스트"
echo "####################################"
echo "" 

CUDA_VISIBLE_DEVICES=0 python train.py --config config.json