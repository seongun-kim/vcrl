# Value Ensemble
import torch.nn as nn

from vcrl.torch.networks import MlpQf


class QfEnsemble(nn.Module):
    def __init__(
        self,
        num_ensemble,
        *args,
        **kwargs
    ):
        super(QfEnsemble, self).__init__()
        self.num_ensemble = num_ensemble
        qfs = []
        for _ in range(num_ensemble):
            qf = MlpQf(
                *args,
                **kwargs
            )
            qfs.append(qf)
        self.qfs = nn.ModuleList(qfs)

    def forward(self, obs, actions):
        out = []
        for qf in self.qfs:
            out.append(qf(obs, actions))
        return out

