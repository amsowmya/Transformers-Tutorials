# Transformers-Tutorials

## Data preprocessing
Regarding preparing your data for a PyTorch model, there are a few options:
- a native PyTorch dataset + dataloader. This is the standard way to prepare data for a PyTorch model, namely by subclassing `torch.utils.data.Dataset`, and then creating a corresponding `DataLoader` (which is a Python generator that allows to loop over the items of a dataset). When subclassing the `Dataset` class, one needs to implement 3 methods: `__init__`, `__len__` (which returns the number of examples of the dataset) and `__getitem__` (which returns an example of the dataset, given an integer index).Here's an exampld of creating a basic text classification dataset (assuming one has a csv that contains 2 columns, namely "text" and "label"):

```python
from torch.util.data import Dataset

class CustomTrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        item = df.iloc[idx]
        text = item['text']
        label = item['label']
        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=128, 
        truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        # add label
        encoding['label'] = torch.tensor(label)

        return encoding
```

Instantiating the dataset then happens as follows:

```python
from transformers import BertTokenizer
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
df = pd.read_csv("path_to_your_csv")

train_dataset = CustomTrainDataset(df=df, tokenizer=tokenizer)
```

Accessing the first example of the dataset can then be done as follows:

```python
encoding = train_dataset[0]
```

In practice, one creates a corresponding `DataLoader`, that allows to get batches from the dataset:

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True
```

I often check whether the data is created correctly by fetching the first batch from the data loader, and then printing out the shapes of the tensors, decoding the input_ids back to text, etc.

```python
batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)
# decode the input_ids of the first example of the batch
print(tokenizer.decode(batch['input_ids'][0].tolist()))
```
-  [HuggingFace Datasets](https://huggingface.co/docs/datasets/). Datasets is a library by HuggingFace that allows to easily load and process data in a very fast and memory-efficient way. It is backed by [Apache Arrow](https://arrow.apache.org/), and has cool features such as memory-mapping, which allow you to only load data into RAM when it is required. It only has deep interoperability with the [HuggingFace hub](https://huggingface.co/datasets), allowing to easily load well-known datasets as well as share your own with the community.

Loading a custom dataset as a Dataset object can be done as follows (you can install datasets using `pip install datasets`):
```python
from datasets import load_dataset

dataset = load_dataset('csv', data_files={'train': ['my_train_file_1.csv', 'my_train_file_2.csv'] 'test': 'my_test_file.csv'})
```
Here I'm loading local CSV files, but there are other formats supported (including JSON, Parquet, txt) as well as loading data from a local Pandas dataframe or dictionary for instance. You can check out the [docs](https://huggingface.co/docs/datasets/loading.html#local-and-remote-files) for all details.

## Training frameworks
Regarding fine-tuning Transformer models (or more generally, PyTorch models), there are a few options:
- using native PyTorch. This is the most basic way to train a model, and requires the user to manually write the training loop. The advantage is that this is very easy to debug. The disadvantage is that one needs to implement training him/herself, such as setting teh model in the appropriate mode (`model.train()`/`model.eval()`), handle divice placement (`model.to(device)`), etc. A typical training loop in PyTorch looks as follows:

```python
import torch
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# I almost always use a learning rate of 5e-5 when fine-tuning Transformer based models
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# put model on GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        # put batch on device
        batch = {k: v.to(device) for k, v in batch.items()}

        # forward pass
        outputs = model(**batch)
        loss = outputs.loss

        train_loss += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Loss after epoch {epoch}: ", train_loss/len(train_dataloader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            # put batch on device
            batch = {k: v.to(device) for k, v in batch.items()}

            # forward pass
            outputs = model(**batch)
            loss = outputs.logits

            val_loss += loss.item()
    print("Validation loss after epoch {epoch}: ", val_loss/len(eval_dataloader))
```

- [PyTorch Lightning (PL)](https://www.pytorchlightning.ai/). PyTorch Lightning is a framework that automates the training loop written above, by abstracting it away in a Trainer object. Users don't need to write the training loop themselves anymore, instead they can just do `trainer = Trainer()` and then `trainer.fit(model)`. The advantage is that you can start training models very quickly (hence the name lightning), as all training-related code is handled by the `Trainer` object. The disadvantage is that it may be more difficult to debug your model, as the training and evaluation is now abstracted away.
- [HuggingFace Trainer](https://huggingface.co/transformers/main_classes/trainer.html). The HuggingFace Trainer API can be seen as a framework similar to PyTorch Lightning in the sense that it also abstracts the training away using a Trainer object. However, contrary to PyTorch Lightning, it is not meant not be a general framework. Rather, it is made especially for fine-tuning Transformer-based models available in the HuggingFace Transformers library. The Trainer also has an extension called `Seq2SeqTrainer` for encoder-decoder models, such as BART, T5 and the `EncoderDecoderModel` classes. Note that all [PyTorch example scripts](https://github.com/huggingface/transformers/tree/master/examples/pytorch) of the Transformers library make use of the Trainer.
- [HuggingFace Accelerate](https://github.com/huggingface/accelerate): Accelerate is a new project, that is made for people who still want to write their own training loop (as shown above), but would like to make it work automatically irregardless of the hardware (i.e. multiple GPUs, TPU pods, mixed precision, etc.).
