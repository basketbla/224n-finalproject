from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BertForSequenceClassification, BertTokenizer
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Load in reddit data
dataset = load_dataset('csv', data_files='total_labeled_nonan.csv')

# Take 11000 random examples for train and 3000 validation
dataset = DatasetDict(
    train=dataset['train'].shuffle(seed=1111).select(range(11000)),
    val=dataset['train'].shuffle(seed=1111).select(range(11000, 14000)),
)

# Do padding and truncation and some stuff I'm not sure about
small_tokenized_dataset = dataset.map(
    lambda example: tokenizer(example['text'], padding=True, truncation=True),
    batched=True,
    batch_size=16
)

small_tokenized_dataset = small_tokenized_dataset.remove_columns(["text"])
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")


# Remove the weird things that only came up for me
small_tokenized_dataset = small_tokenized_dataset.remove_columns(['__index_level_0__'])
small_tokenized_dataset = small_tokenized_dataset.remove_columns(['token_type_ids'])


# Make these dataloader things
train_dataloader = DataLoader(small_tokenized_dataset['train'], batch_size=16)
eval_dataloader = DataLoader(small_tokenized_dataset['val'], batch_size=16)


# Do the actual training
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

num_epochs = 3
num_training_steps = 3 * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

best_val_loss = float("inf")
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    # training
    model.train()
    for batch_i, batch in enumerate(train_dataloader):
        
        output = model(**batch)
        
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)
    
    # validation
    model.eval()

    for batch_i, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            output = model(**batch)
        loss += output.loss
    
    avg_val_loss = loss / len(eval_dataloader)
    print(f"Validation loss: {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        print("Saving checkpoint!")
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            },
            f"checkpoints/epoch_{epoch}.pt"
        )  