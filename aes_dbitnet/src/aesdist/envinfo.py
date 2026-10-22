"""Environment/hardware reporting and strict device selection. Never edits CUDA/SLURM config."""
import os
import platform
import sys

SLURM_KEYS = ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS",
              "SLURM_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE",
              "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS",
              "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"]


def configure_device(mode, threads=None):
    """mode: 'cpu' hides GPUs from TensorFlow (framework-level, env untouched).
    'gpu' requires a SLURM allocation AND a GPU visible to TensorFlow, else raises.
    Must be called before TensorFlow creates any context."""
    import tensorflow as tf
    if threads is None:
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None
    if threads:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, min(2, threads)))
    if mode == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return {"device": "cpu", "threads": threads}
    if mode == "gpu":
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("GPU mode requires running inside a SLURM allocation (SLURM_JOB_ID not set). "
                               "Refusing to use a GPU that merely happens to be visible.")
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("GPU mode requested but TensorFlow sees no GPU in this allocation. "
                               "Check the administrator-approved GPU request syntax; not falling back to CPU.")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        return {"device": "gpu", "gpus": [g.name for g in gpus], "threads": threads}
    raise ValueError(mode)


def collect_env(include_tf=True):
    info = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "hostname": platform.node(), "cpu_count": os.cpu_count(),
            "slurm_env": {k: os.environ.get(k) for k in SLURM_KEYS if os.environ.get(k) is not None}}
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception as e:
        info["numpy"] = f"unavailable: {e}"
    if include_tf:
        try:
            import tensorflow as tf
            info["tensorflow"] = tf.__version__
            info["keras"] = getattr(tf.keras, "__version__", "?")
            info["tf_visible_gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
            info["tf_built_with_cuda"] = tf.test.is_built_with_cuda()
        except Exception as e:
            info["tensorflow"] = f"unavailable: {e}"
    try:
        with open("/proc/cpuinfo") as f:
            names = [l.split(":", 1)[1].strip() for l in f if l.startswith("model name")]
        info["cpu_model"] = names[0] if names else None
    except Exception:
        pass
    return info
