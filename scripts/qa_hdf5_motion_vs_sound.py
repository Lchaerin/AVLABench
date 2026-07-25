import h5py, numpy as np, glob, json, sys

files = sorted(glob.glob('dataset_dooropen/take_out_microwave_food/data_*.hdf5'))
mo, ao, qd = [], [], []
for hf in files:
    try:
        f = h5py.File(hf, 'r')
        g = list(f['data'].values())[0]
        q = np.array(g['observation/q_state'])
        dev = np.max(np.abs(q - q[0]), axis=1)
        m = int(np.argmax(dev > 0.05)) if (dev > 0.05).any() else -1
        oa = json.loads(g['meta_info/oracle_audio'][()])
        afe = oa['active_sources'][0]['active_from_frame']
        f.close()
        if m >= 0:
            mo.append(m); ao.append(afe); qd.append(m - afe)
    except Exception as e:
        print('skip', hf, e)
mo, ao, qd = np.array(mo), np.array(ao), np.array(qd)
print('episodes:', len(qd))
print('audio active_from_frame: mean %.1f min %d max %d' % (ao.mean(), ao.min(), ao.max()))
print('q_state motion onset frame: mean %.1f min %d max %d' % (mo.mean(), mo.min(), mo.max()))
print('motion-audio (frames): mean %.1f median %.1f' % (qd.mean(), np.median(qd)))
print('fraction moved AFTER audio:', float((qd >= 0).mean()))
