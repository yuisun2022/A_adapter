# Adversarial training with adapter in NLP/NLU

Adversarial training and adapter structure benefit pre-trained model transfer learning, especially in low-resource datasets. Here we upload different kinds of adversarial training algorithms w/o adapters for easily modify. *A_adapter* includes several adversarial algorithms with adapter pre-trained models. *Adversarial Training for NLU* includes the same but without the adapter.

## Instructions
This work is based on [huggingface/transformers](https://github.com/huggingface/transformers) and [adapter-Hub](https://github.com/adapter-hub/adapter-transformers).
Notice that w/o adapter does **NOT** share the same environment. It would be best if builts them separately.

1. Set up
```
$ pip install -r requirements.txt
```
2. Training
```
export TASK_NAME=cola
python run_glue_Aadapter.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 7e-4 \
  --num_train_epochs 10
```
3. Using other pre-trained model
If you want to change for other pre-trained like RoBERTa, don't forget to give the **base_model** name to the adversarial training class.
```
# run_glue_XXX.py

# init adversarial class
adv = Aadapter(adv_K=3, adv_lr=1e-1, adv_init_mag=2e-2, adv_max_norm=1.0, adv_norm_type='l2', base_model='roberta')
```

### computation cost
Memory:
We use single GPU(NVIDIA GeForce RTX 3090) and 20~22G RAM in our setting. The batch size and sentence length depend on your device.
Time:
|   Task   |  Metric  | Training time |
| -------- | -------- | ------------- |
|   CoLA   | Matthews corr. | 3 min |
|   SST-2  | Accuracy | 30 min |
|   MRPC   | F1/Accuracy | 3 min |
|   STS-B  | Pearson/Spearman corr. | 4 min |
|   QQP    | Accuracy/F1 | 4 hr 21 min |
|   MNLI   | Matched acc./Mismatched acc. | 5 hr 46 min |
|   QNLI   | Accuracy | 1 hr 40 min |
|   RTE    | Accuracy | 3 min |
|   WNLI   | Accuracy | 30 sec |
