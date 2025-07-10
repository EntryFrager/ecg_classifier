def get_stat(dataset, target_labels):
    for key, _ in target_labels.items():
        label = dataset[key].value_counts()
        print(f'{key} unique labels: {label}')