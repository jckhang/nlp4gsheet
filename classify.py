from typing import List, Tuple, Dict
import pandas as pd
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

df = pd.DataFrame(
    {
        "text": [
            "Lenovo announced a new partnership with a major telecommunications company.",
            "The new Lenovo Galaxy S22 was released today with a faster processor and improved camera.",
            "Lenovo employees volunteered at a local food bank to help those in need.",
            "A class-action lawsuit was filed against Lenovo alleging defects in their washing machines.",
            "Lenovo announced a new initiative to reduce their carbon footprint by 50% over the next decade.",
            'The Lenovo Gear Sport smartwatch was named "Best in Class" by a leading tech magazine.',
            "Lenovo hosted a charity auction for children's hospitals, raising over $1 million.",
            "Lenovo faced backlash from consumers after a controversial advertisement for their latest phone.",
            "The Lenovo Galaxy Note 10 is rumored to have a larger screen and improved stylus.",
            "Lenovo employees protested outside company headquarters, demanding better working conditions.",
            "Samsung employees protested outside company headquarters, demanding better working conditions.",
            "Some people protested outside company headquarters, demanding better working conditions.",
            "I love cats and dogs.",
        ],
        "tags": [
            "corporate news",
            "product news",
            "social news",
            "corporate news",
            "corporate news",
            "product news",
            "social news",
            "corporate news",
            "product news",
            "social news",
            "not related",
            "not related",
            "not related",
        ],
    }
)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def one_hot(labels: List[str]) -> Tuple[List[str], Dict[int, str]]:
    tags_map = {}
    tags = []
    for i, tag in enumerate(labels):
        if tag not in tags_map:
            tags_map[tag] = len(tags_map)
        tags.append(tags_map[tag])
    return tags, {v: k for k, v in tags_map.items()}


def run_classifier(examples: List[Dict[str, str]], input_value: str) -> str:
    df = pd.DataFrame(examples)
    texts = df["text"].values.tolist()
    tags, tags_map = one_hot(df["tags"].values.tolist())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, tags, test_size=0.2
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(tags_map)
    )
    model.to(device)
    model.train()
    epoch = 10

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=epoch,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
        learning_rate=3e-5,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
    )

    trainer.train()

    test_encodings = tokenizer([input_value], truncation=True, padding=True)
    prediction = model(
        torch.tensor(test_encodings["input_ids"]).to(device),
        attention_mask=torch.tensor(test_encodings["attention_mask"]).to(device),
    )
    clz = prediction[0].argmax().item()
    print(f"Predicted class: {tags_map[clz]}")
