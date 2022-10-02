import os
import os.path as osp
import torch

from .distributed import is_main_process


class Checkpointer(object):

    def __init__(self, path, max_n=3):
        self.path = path
        self.max_n = max_n
        self.checkpointables = {}

        self.start_epoch = 1
        self.start_iter = 1
        self.epoch = 1
        self.iter = 1
        self.best_iter = -1
        self.best_epoch = -1
        self.best_score = -1e16
        self.fns = []

    def _rm(self, path):
        if osp.isfile(path):
            os.remove(path)

    def __len__(self):
        return len(self.fns)

    def add_model(self, model, opt=None):
        self.checkpointables = {}
        self.checkpointables['model'] = model
        self.checkpointables['optimizer'] = opt

    def clean(self, n_remove):

        n_remove = min(n_remove, len(self.fns))

        for i in range(n_remove):
            fn = self.fns[i]
            for name, data in self.checkpointables.items():
                path = osp.join(self.path, f'{name}_{fn}.pth')
                self._rm(path)

        removed = self.fns[:n_remove]
        self.fns = self.fns[n_remove:]
        return removed

    def load(self, path):
        state_dict = torch.load(path, map_location='cpu')
        self.start_epoch = state_dict['epoch'] + 1
        self.start_iter = state_dict['iter'] + 1
        self.best_epoch = state_dict['epoch']
        self.best_iter = state_dict['iter']
        self.best_score = state_dict['score']

        for name, model in self.checkpointables.items():
            model.load_state_dict(state_dict[name])

        return True

    # def find(self, suffix, force=False):
    #     paths = {}
    #     found = True
    #     for name, data in self.models.items():
    #         paths[name] = {}
    #         for d in ('model', 'opt'):
    #             fn = self._filename(d, name, suffix)
    #             path = self._get_full_path(fn)
    #             paths[name][d] = path
    #             if not os.path.isfile(path):
    #                 print("File not found: ", path)
    #                 if d == 'model':
    #                     found = False

    #     if found and suffix not in self.checkpoints:
    #         if len(self.checkpoints) < self.max_n or force:
    #             self.checkpoints.insert(0, suffix)
    #             if force:
    #                 self.max_n = max(self.max_n, len(self.checkpoints))

    #     return found, paths

    def checkpoint(self, epoch, iter, score):
        if is_main_process():
            fn = f'epoch{epoch:03d}_iter{iter:07d}_score{score:4.3f}'
            self.fns.append(fn)

            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self.best_iter = iter

            save_dict = {}
            for name, data in self.checkpointables.items():
                if data is not None:
                    save_dict[name] = data.state_dict()

            save_dict['epoch'] = epoch
            save_dict['iter'] = iter
            save_dict['score'] = score

            path = osp.join(self.path, f'{fn}.pth')
            if not osp.isfile(path):
                torch.save(save_dict, path)

            # removing
            n_remove = max(0, len(self.fns) - self.max_n)
            removed = self.clean(n_remove)

            return removed
