BERT_BASE_DIR="/home/aufish/Downloads/bert"

python3 -m bert.run_squad\
    --vocab_file=$BERT_BASE_DIR/assets/vocab.txt \
    --init_checkpoint=$BERT_BASE_DIR/saved_model.pb\
    --output_dir=/tmp/squad_base/
