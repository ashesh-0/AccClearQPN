from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded


class Grid:
    def __init__(self, img_size, total_grid, ith_grid, pad_grid):
        self._img_size = img_size
        self._start_x = None
        self._start_y = None
        self._N = total_grid
        self.ith = ith_grid
        self._pad = pad_grid
        self._multiple = 30
        self._init_grid()
        assert self.ith >= 0 and self.ith < self._N * self._N
        print(f'[{self.__class__.__name__}] N:{self._N} I:{self.ith} P:{self._pad} Imgsz:{self._img_size}]')

    def _init_grid(self):
        Nx, Ny = self._img_size
        self._start_x = [i * Nx // self._N for i in range(self._N)]
        self._start_y = [i * Ny // self._N for i in range(self._N)]
        self._start_x.append(Nx)
        self._start_y.append(Ny)

    def get_total_padding(self, sx, ex, sy, ey):
        idx_x = self.ith // self._N
        idx_y = self.ith % self._N
        lpadx = self._start_x[idx_x] - sx
        rpadx = ex - self._start_x[idx_x + 1]

        lpady = self._start_y[idx_y] - sy
        rpady = ey - self._start_y[idx_y + 1]
        return lpadx, rpadx, lpady, rpady

    def _get_grid_dim(self, dim, start_arr, idx):
        s = max(0, start_arr[idx] - self._pad)
        e = min(dim, start_arr[idx + 1] + self._pad)
        diff = (self._multiple - (e - s) % self._multiple) % self._multiple
        #         print(s,e,diff,dim)
        if s == 0:
            s_final = s
            e_final = e + diff
        elif e == dim:
            s_final = s - diff
            e_final = e
        else:
            s_final = s - diff // 2
            e_final = e + (diff - (s - s_final))

        assert s_final >= 0 and e_final <= dim, f'{s_final}-{e_final}-{dim}'
        assert (e_final - s_final) % self._multiple == 0
        return s_final, e_final

    def get_grid_bbox(self):
        Nx, Ny = self._img_size
        idx_x = self.ith // self._N
        idx_y = self.ith % self._N
        sx, ex = self._get_grid_dim(Nx, self._start_x, idx_x)
        sy, ey = self._get_grid_dim(Ny, self._start_y, idx_y)
        return sx, ex, sy, ey

    def get_padding_bbox(self):
        sx, ex, sy, ey = self.get_grid_bbox()
        return self.get_total_padding(sx, ex, sy, ey)


class DataLoaderGrid(DataLoaderAllLoaded):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k not in ['total_grid', 'ith_grid', 'pad_grid']})
        assert self._residual is False
        N = kwargs.get('total_grid', 3)
        ith = kwargs['ith_grid']
        pad = kwargs.get('pad_grid', 10)
        img_size = kwargs['img_size']
        self._grid = Grid(img_size, N, ith, pad)

    def get_info_for_model(self):
        model_dict = super().get_info_for_model()
        model_dict['padding_bbox'] = self._grid.get_padding_bbox()
        return model_dict

    def __getitem__(self, input_index):
        inp, target, mask = super().__getitem__(input_index)
        sx, ex, sy, ey = self._grid.get_grid_bbox()
        return inp[:, :, sx:ex, sy:ey], target[:, sx:ex, sy:ey], mask[:, sx:ex, sy:ey]
