from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import os

def run_tensorboard(log_dir, port):
    os.system(f"tensorboard --logdir={log_dir} --port={port}")

class TensorBoardLogger:
    def __init__(self, log_dir='runs', enabled=True):
        self.enabled = enabled
        if self.enabled:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def log_scalar(self, tag, value, step):
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag, img_tensor, step):
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)


    def launch_tensorboard(self, port=6006):
        """
        TensorBoard 서버를 시작합니다. 별도의 프로세스에서 TensorBoard를 실행합니다.
        로깅이 활성화된 경우에만 실행됩니다.
        localhost:6006/
        """
        if self.enabled and self.writer:
            tensorboard_process = multiprocessing.Process(target=run_tensorboard, args=(self.writer.log_dir, port))
            tensorboard_process.start()

    def close(self):
        if self.enabled and self.writer is not None:
            self.writer.close()
