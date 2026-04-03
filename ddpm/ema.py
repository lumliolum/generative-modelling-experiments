# in this we will implement model ema wrapper that we will use
# reference : https://github.com/ermongroup/ncsnv2/blob/master/models/ema.py
# https://medium.com/@heyamit10/exponential-moving-average-ema-in-pytorch-eb8b6f1718eb


class EMAHelper:
    def __init__(self, mu, model):
        self.mu = mu
        self.model = model
        self.shadow = {}
        self.backup = {}

        # register
        self.register(model)

    def register(self, module):
        """
            For our use case, module means model. This function should be
            used for initialization. all model parameters will be stored for the first
            time in this function
        """
        for name, param in module.named_parameters():
            # only store the params the has requires grad to be true
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
            Update the exponential average between shadow and given module. The update is given by
            (1 - mu) * new weight + mu * old weight
        """
        for name, param in self.model.named_parameters():
            self.shadow[name].data = (1 - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # take the backup
                self.backup[name] = param.data.clone()
                # apply the exponential average
                param.data = self.shadow[name]

    def restore(self):
        for name,  param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
