from maml_zoo.logger import logger
import pyprind

class ProgBarCounter(object):
    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        self.pbar = pyprind.ProgBar(self.max_progress)


    def inc(self, increment):
        self.cur_count += increment
        new_progress = self.cur_count * self.max_progress / self.total_count
        if new_progress < self.max_progress:
            self.pbar.update(new_progress - self.cur_progress)
        self.cur_progress = new_progress

    def stop(self):
        if self.pbar is not None and self.pbar.active:
            self.pbar.stop()
