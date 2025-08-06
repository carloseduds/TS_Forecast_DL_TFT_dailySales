import mlflow
import psutil
import torch
from lightning.pytorch.callbacks import Callback


# Callback: Loga todas as métricas do Lightning por época no MLflow
class MLflowMetricsLogger(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        for metric_name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            try:
                mlflow.log_metric(metric_name, value, step=epoch)
            except Exception as e:
                pass  # ignora métricas não logáveis


# Callback: Loga uso de CPU e RAM no MLflow por época
class SystemUsageLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        try:
            mlflow.log_metric("cpu_percent", psutil.cpu_percent(), step=epoch)
            mlflow.log_metric("memory_percent", psutil.virtual_memory().percent, step=epoch)
        except Exception as e:
            pass


# Callback: Loga uso de GPU (NVIDIA) no MLflow por época (usa GPUtil)
try:
    import GPUtil

    class GPUUsageLogger(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            try:
                for gpu in GPUtil.getGPUs():
                    mlflow.log_metric(f"gpu_{gpu.id}_mem_used_mb", gpu.memoryUsed, step=epoch)
                    mlflow.log_metric(f"gpu_{gpu.id}_load", gpu.load, step=epoch)
            except Exception as e:
                pass

except ImportError:
    # Se não tem GPUtil, define classe dummy para não quebrar import
    class GPUUsageLogger(Callback):
        pass
