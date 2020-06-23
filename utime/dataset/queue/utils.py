from utime.utils import ensure_list_or_tuple
from utime.dataset.queue import (StudyLoader, LimitationQueue,
                                 LazyQueue, EagerQueue)


QUEUE_TYPE_TO_CLS = {
    "limitation": LimitationQueue,
    "lazy": LazyQueue,
    'eager': EagerQueue
}


def get_dataset_queues(datasets,
                       queue_type,
                       n_load_threads=7,
                       logger=None,
                       **kwargs):
    if datasets is None:
        return None
    datasets = ensure_list_or_tuple(datasets)

    # Prepare study loader object
    max_loaded = kwargs.get("max_loaded_per_dataset", 0) * len(datasets)
    study_loader = StudyLoader(n_threads=n_load_threads,
                               max_queue_size=max_loaded or None,
                               logger=logger)

    # Get a queue for each dataset
    queues = []
    queue_cls = QUEUE_TYPE_TO_CLS[queue_type.lower()]
    for dataset in datasets:
        queue = queue_cls(
            dataset=dataset,
            max_loaded=kwargs.get("max_loaded_per_dataset"),
            study_loader=study_loader,
            logger=logger,
            **kwargs
        )
        queues.append(queue)
    return queues
