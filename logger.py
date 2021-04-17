from argparse import Namespace
from time import time
from typing import Optional, Dict, Any, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no-cover
    mlflow = None
    MlflowClient = None
    _MLFLOW_AVAILABLE = False


from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


LOCAL_FILE_URI_PREFIX = "file:"

class MLFlowLogger2(LightningLoggerBase):

    def __init__(self,
                 experiment_name: str = 'default',
                 run_name: str = 'test',
                 tracking_uri: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 save_dir: Optional[str] = './mlruns'):

        if not _MLFLOW_AVAILABLE:
            raise ImportError('You want to use `mlflow` logger which is not installed yet,'
                              ' install it with `pip install mlflow`.')
        super().__init__()
        if not tracking_uri:
            tracking_uri = f'{LOCAL_FILE_URI_PREFIX}{save_dir}'

        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self.tags = tags

        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        self._experiment_id = run.info.experiment_id
        self._mlflow_client = MlflowClient(tracking_uri)

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        return self._mlflow_client

    @property
    def run_id(self):
        # create the experiment if it does not exist to get the run id
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self):
        # create the experiment if it does not exist to get the experiment id
        _ = self.experiment
        return self._experiment_id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for k, v in params.items():
            self.experiment.log_param(self.run_id, k, v)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning(f'Discarding metric with string value {k}={v}.')
                continue
            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step)

    @rank_zero_only
    def finalize(self, status: str = 'FINISHED') -> None:
        super().finalize(status)
        status = 'FINISHED' if status == 'success' else status
        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)

    @property
    def save_dir(self) -> Optional[str]:
        if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
            return self._tracking_uri.lstrip(LOCAL_FILE_URI_PREFIX)

    @property
    def name(self) -> str:
        return self.experiment_id

    @property
    def version(self) -> str:
        return self.run_id
