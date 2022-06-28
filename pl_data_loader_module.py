from datetime import datetime

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from core.data_loader_type import DataLoaderType
from core.enum import DataType
from data_loaders.classifier.classifier_data_loader import ClassifierDataLoader
from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded
from data_loaders.data_loader_grid import DataLoaderGrid
from data_loaders.data_loader_with_prior import DataLoaderWithPrior


class PLDataLoader(LightningDataModule):
    def __init__(
        self,
        train_start: datetime,
        train_end: datetime,
        val_start: datetime,
        val_end: datetime,
        data_type=DataType.NoneAtAll,
        dloader_type=None,
        input_len=None,
        target_len=None,
        target_avg_length=6,
        target_offset=0,
        threshold=None,
        hourly_data=False,
        residual=False,
        img_size=None,
        prior_dtype=DataType.NoneAtAll,
        ith_grid=-1,
        pad_grid=None,
        random_std=0,
        sampling_rate=None,
        batch_size=32,
        num_workers=4,
    ):

        super().__init__()
        self._tr_s = train_start
        self._tr_e = train_end
        self._val_s = val_start
        self._val_e = val_end
        self._ilen = input_len
        self._tlen = target_len
        self._tavg_len = target_avg_length
        self._toffset = target_offset
        self._workers = num_workers
        self._img_sz = img_size
        self._threshold = threshold
        self._batch_size = batch_size
        self._dtype = data_type
        self._hourly_data = hourly_data
        self._residual = residual
        self._prior = prior_dtype
        self._sampling_rate = sampling_rate
        self._random_std = random_std
        self._dloader_type = dloader_type
        self._ith_grid = ith_grid
        self._pad_grid = pad_grid
        self._train_dataset = None
        self._val_dataset = None
        self._setup()
        self.model_related_info = self._train_dataset.get_info_for_model()

    # def setup(self, stage=None):
    #     pass

    def _setup(self):
        if self._dloader_type == DataLoaderType.WithPrior:
            self._train_dataset = DataLoaderWithPrior(
                self._tr_s,
                self._tr_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                prior_dtype=self._prior,
                random_std=self._random_std,
                is_train=True,
            )

            self._val_dataset = DataLoaderWithPrior(
                self._val_s,
                self._val_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                prior_dtype=self._prior,
                is_validation=True,
            )

        elif self._dloader_type == DataLoaderType.Native:
            self._train_dataset = DataLoaderAllLoaded(
                self._tr_s,
                self._tr_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                random_std=self._random_std,
                is_train=True,
            )

            self._val_dataset = DataLoaderAllLoaded(
                self._val_s,
                self._val_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                is_validation=True,
            )
        elif self._dloader_type == DataLoaderType.Classifier:
            self._train_dataset = ClassifierDataLoader(
                self._tr_s,
                self._tr_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                random_std=self._random_std,
                is_train=True,
            )

            self._val_dataset = ClassifierDataLoader(
                self._val_s,
                self._val_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                is_validation=True,
            )
        elif self._dloader_type == DataLoaderType.Grid:
            self._train_dataset = DataLoaderGrid(
                self._tr_s,
                self._tr_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                ith_grid=self._ith_grid,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                random_std=self._random_std,
                pad_grid=self._pad_grid,
                is_train=True,
            )

            self._val_dataset = DataLoaderGrid(
                self._val_s,
                self._val_e,
                self._ilen,
                self._tlen,
                target_avg_length=self._tavg_len,
                target_offset=self._toffset,
                threshold=self._threshold,
                data_type=self._dtype,
                hourly_data=self._hourly_data,
                residual=self._residual,
                ith_grid=self._ith_grid,
                img_size=self._img_sz,
                sampling_rate=self._sampling_rate,
                pad_grid=self._pad_grid,
                is_validation=True,
            )

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, num_workers=self._workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._batch_size, num_workers=self._workers, shuffle=False)
