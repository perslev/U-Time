

from utime.dataset import SleepStudyDataset
from utime.dataset.queue import StudyLoader, LimitationQueue, LazyQueue, EagerQueue
from utime.dataset.queue.utils import get_dataset_queues


def get_dataset():
    anot = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
        'Sleep stage ?': 5,
        'Movement time': 5
    }
    ssd = SleepStudyDataset(data_dir='/Users/mathiasperslev/OneDrive/University/phd/'
                                     'data/sleep/multi_dataset_standard_formatting/'
                                     'sedf_sc',
                            folder_regex='SC.*',
                            annotation_dict=anot)
    ssd.set_select_channels(['Fpz-Cz'])
    ssd.set_strip_func('drop_class', class_int=5)
    ssd.set_scaler("RobustScaler")
    ssd.set_quality_control_func("clip_noisy_values")
    return ssd


datasets = [get_dataset() for _ in range(4)]
for i, d in enumerate(datasets):
    d._identifier = i

queues = get_dataset_queues(datasets,
                            queue_type='limitation',
                            max_loaded_per_dataset=50,
                            num_access_before_reload=50,
                            await_preload=False)
queues[0].study_loader.join()

import time

for i in range(100000):
    loaded, not_loaded, load_queue = [], [], []
    for queue in queues:
        with queue.get_random_study() as ss:
            loaded.append(queue.loaded_queue.qsize())
            not_loaded.append(queue.non_loaded_queue.qsize())
            load_queue.append(queue.study_loader.qsize())
            time.sleep(0.0025)
    print(loaded)
    print(not_loaded)
    print(load_queue)
    print("--")

# with queue.get_study_by_id('SC4071E0') as study:
#     print(study)
# with queue.get_study_by_idx(0) as study:
#     print(study)  # SC4072E0

# while True:
#     print("here")
#     with queue.get_random_study() as study:
#         print(study)
