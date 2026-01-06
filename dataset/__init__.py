import importlib

def find_dataset(dataset):
    if dataset == "rpc":
        name = 'dataset.dataset_rpc'
        module = importlib.import_module(name)
    elif dataset == "pinhole":
        name = 'dataset.dataset_pinhole'
        module = importlib.import_module(name)
    else:
        raise Exception(f"expect 'rpc' or 'pinhole', but receive: {dataset}")

    return getattr(module, "MVSDataset")