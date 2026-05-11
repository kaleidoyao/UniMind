from pathlib import Path
from shock.utils import h5Dataset
from shock.utils import preprocessing_cnt, preprocessing_gdf, preprocessing_edf, preprocessing_mat

savePath = Path('/home/bingxing2/ailab/wangjunyu/LaBraM/LaBraM-main/dataset/emotion/SEED')
rawDataPath = Path('/home/bingxing2/ailab/group/ai4neuro/UniMind/SEED/data_split_to_1s')
datatype = "mat"
useless_chs = []
group = rawDataPath.glob('*.' + datatype)

# preprocessing parameters
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

# channel number * rsfreq
chunks = (62, rsfreq)

dataset = h5Dataset(savePath, 'dataset')
for cntFile in group:
    print(f'processing {cntFile.name}')
    if datatype == 'gdf':
        eegData, chOrder = preprocessing_gdf(cntFile, l_freq, h_freq, rsfreq, drop_channels=useless_chs)
    elif datatype == 'cnt':
        eegData, chOrder = preprocessing_cnt(cntFile, l_freq, h_freq, rsfreq)
    elif datatype == 'edf':
        eegData, chOrder = preprocessing_edf(cntFile, l_freq, h_freq, rsfreq)
    elif datatype == 'mat':
        eegData, chOrder = preprocessing_mat(cntFile, l_freq, h_freq, rsfreq)
    else:
        print("unknown datatype...")
        exit(0)
    
    if datatype != 'mat':
        chOrder = [s.upper() for s in chOrder]
    eegData = eegData[:, :-10*rsfreq]
    grp = dataset.addGroup(grpName=cntFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
