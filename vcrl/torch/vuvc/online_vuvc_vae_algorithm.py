import gtimer as gt
from vcrl.core import logger

from vcrl.data_management.online_vuvc_vae_replay_buffer import OnlineVuvcVaeRelabelingBuffer
from vcrl.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm


class OnlineVuvcVaeAlgorithm(OnlineVaeAlgorithm):

    def __init__(
            self,
            qfe,
            qfe_trainer,            
            *base_args,
            **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs)
        assert isinstance(self.replay_buffer, OnlineVuvcVaeRelabelingBuffer)

        self.qfe = qfe
        self.qfe_trainer = qfe_trainer

    def _train(self):
        super()._train()
        self._cleanup()

    def _end_epoch(self, epoch):
        self._train_qfe(epoch)
        gt.stamp('qfe training')
        super()._end_epoch(epoch)

    def _log_stats(self, epoch):
        self._log_qfe_stats()
        super()._log_stats(epoch)

    def to(self, device):
        for net in self.qfe_trainer.networks:
            net.to(device)
        super().to(device)

    def training_mode(self, mode):
        for net in self.qfe_trainer.networks:
            net.train(mode)
        super().training_mode(mode)

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        assert 'qfe' not in snapshot
        snapshot['qfe'] = self.qfe
        return snapshot

    def _train_qfe(self, epoch):
        for _ in range(self.num_train_loops_per_epoch):
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.qfe_trainer.train(train_data)
            self.training_mode(False)

    def _log_qfe_stats(self):
        logger.record_dict(
            self.qfe_trainer.get_diagnostics(),
            prefix='qfe_trainer/',
        )
