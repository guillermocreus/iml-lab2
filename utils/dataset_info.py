import numpy as np

def dataset_info(data, y, dataset_name=''):
    print('_'*20)
    print(f'Information of dataset {dataset_name}')
    print(f'Number of instances: {data.shape[0]}')
    print(f'Number of features: {data.shape[1]}')
    n_labels = 0
    for num in np.unique(y.to_numpy()):
        n_label_int = (y == num).sum()
        n_labels += n_label_int
        print(f'Number of instances of class {num}: {n_label_int}')

    print(f'Number of labels: {n_labels}')
    print(f'Number of instances after preprocessing {data.shape[0]}')
    print('_' * 20)