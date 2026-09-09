def split_dataset(dataset):
    print("Dataset splits:", dataset.keys())
    if "test" in dataset:
        return dataset["train"], dataset["test"]

    if "validation" in dataset:
        return dataset["train"], dataset["validation"]

    split = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42,
    )

    return split["train"], split["test"]

def print_dataset_statistics(train_dataset):
    total = 0
    count = 0
    
    for sample in train_dataset:
    
        valid = sum(1 for x in sample["labels"] if x != -100)
        total += valid
        count += 1
    
    if count > 0:
        print("Average label tokens:", total / count)
