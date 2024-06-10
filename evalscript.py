from datasets import load_dataset, get_dataset_config_names
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import evaluate
import argparse 
from tqdm import tqdm

SEED = 42
NUM_PROC=5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
CACHE=None

def dataloader_evaluate(dataset_name, label):
  lang_accuracies = {}
  #clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
  clf_metrics=evaluate.load("accuracy")
  langs = get_dataset_config_names(f"mbzuai-ugrip-statement-tuning/{dataset_name}")
  print(langs)

  for lang_code in langs:
    print(f"Processing {lang_code}...")
    predictions = []
    actual_labels = []

    dataset = load_dataset(f'mbzuai-ugrip-statement-tuning/{dataset_name}', lang_code, split='test', cache_dir=CACHE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    statement_fields = dataset.column_names # ['statement1', 'statement2', 'statement3']
    statement_fields.remove(label)

    for batch in tqdm(dataloader):
      probabilities = []
      # generate 'prob1', 'prob2', 'prob3'
      for statement_field in statement_fields:
        tok = tokenizer(batch[statement_field], return_tensors='pt', padding=True).to(device)
        probabilities.append(F.softmax(model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).logits, dim=-1)[:,1])
        # ground truth
      labels = batch[label]
      preds = torch.argmax(torch.stack(probabilities, dim=-1), dim=-1)
      predictions.extend(preds.cpu().tolist())
      actual_labels.extend(labels.cpu().tolist())
    # computing macro f1
    lang_accuracies[lang_code] = f"{clf_metrics.compute(predictions=predictions, references=actual_labels)['accuracy']*100:.2f}"

  return lang_accuracies

if __name__ == "__main__":
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument("--model", help='HuggingFace path to model')
  parser.add_argument("--tokenizer", help='HuggingFace path to tokenizer')
  parser.add_argument("--cache", help='cachedir for HuggingFace datasets', default=None)
  
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = AutoModelForSequenceClassification.from_pretrained(args.model).eval().to(device)
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

  print(f"XCOPA Macro F1: {dataloader_evaluate('xcopa', 'label')}")
  print(f"XNLI Macro F1: {dataloader_evaluate('xnli', 'label')}")
  print(f"XWinograd Macro F1: {dataloader_evaluate('xwinograd', 'answer')}")
  print(f"XStoryCloze Macro F1: {dataloader_evaluate('xstorycloze', 'answer_right_ending')}")
