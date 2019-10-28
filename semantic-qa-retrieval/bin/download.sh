#wget https://msmarco.blob.core.windows.net/msmsarcov1/train_v1.1.json.gz -O MSMARCO_train_v1.1.json.gz
#gunzip MSMARCO_train_v1.1.json.gz

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O SQuAD_train_v1.1.json 
cat SQuAD_train_v1.1.json | jq '.data|.[0]|{"data":[.]}' > sample_squad.json

