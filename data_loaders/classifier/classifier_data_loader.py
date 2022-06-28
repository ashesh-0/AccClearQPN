from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded


class ClassifierDataLoader(DataLoaderAllLoaded):
    def __getitem__(self, input_index):
        inp, target, mask = super().__getitem__(input_index)
        target = (target > 1).astype(target.dtype)
        return inp, target, mask
