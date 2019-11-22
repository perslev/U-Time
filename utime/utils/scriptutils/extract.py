

def to_h5_file(out_path, data, channel_names, sample_rate, date, **kwargs):
    """
    Saves a NxC ndarray 'data' of PSG data (N samples, C channels) to a .h5
    archive at path 'out_path'. A list 'channel_names' of length C must be
    passed, giving the name of each channel in 'data'. Each Nx1 array in 'data'
    will be stored under groups in the h5 archive according to the channel name

    Also sets h5 attributes 'date' and 'sample_rate'.

    Args:
        out_path:      (string)   Path to a h5 archive to write to
        data:          (ndarray)  A NxC shaped ndarray of PSG data
        channel_names: (list)     A list of C strings giving channel names for
                                  all channels in 'data'
        sample_rate:   (int)      The sample rate of the signal in 'data'.
        date:          (datetime) A datetime object. Is stored as a timetuple
                                  within the archive. If a non datetime object
                                  is passed, this will be stored 'as-is'.
        **kwargs:
    """
    import h5py
    import time
    from datetime import datetime
    if len(data.shape) != 2:
        raise ValueError("Data must have exactly 2 dimensions, "
                         "got shape {}".format(data.shape))
    if data.shape[-1] == len(channel_names):
        assert data.shape[0] != len(channel_names)  # Should not happen
        data = data.T
    elif data.shape[0] != len(channel_names):
        raise ValueError("Found inconsistent data shape of {} with {} select "
                         "channels ({})".format(data.shape,
                                                len(channel_names),
                                                channel_names))
    if isinstance(date, datetime):
        # Convert datetime object to TS naive unix time stamp
        date = time.mktime(date.timetuple())
    with h5py.File(out_path) as out_f:
        out_f.create_group("channels")
        for chan_dat, chan_name in zip(data, channel_names):
            out_f["channels"][chan_name] = chan_dat
        out_f.attrs['date'] = date
        out_f.attrs["sample_rate"] = sample_rate
