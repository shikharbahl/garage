from garage.tf.algos import NPO
from garage.tf.algos.npo import PGLoss
from garage.tf.optimizers import FirstOrderOptimizer


class VPG(NPO):
    """
    Vanilla Policy Gradient.
    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        if optimizer is None:
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = dict()
        super(VPG, self).__init__(
            pg_loss=PGLoss.Vanilla,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name="VPG",
            **kwargs)
