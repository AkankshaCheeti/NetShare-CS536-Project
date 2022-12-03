make clean && make
./pcapParser ../data/1M/botnet_benign/raw.pcap ../data/1M/botnet_benign/raw.csv
python3 word2vec_embedding.py --src_dir ../data/1M/botnet_benign --word_vec_size 10 --file_type PCAP
python3 preprocess_by_type.py --src_dir ../data/1M/botnet_benign --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header

make clean && make
./pcapParser ../data/1M/botnet_malicious/raw.pcap ../data/1M/botnet_malicious/raw.csv
python3 word2vec_embedding.py --src_dir ../data/1M/botnet_malicious --word_vec_size 10 --file_type PCAP
python3 preprocess_by_type.py --src_dir ../data/1M/botnet_malicious --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header
