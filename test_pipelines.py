# PSTHs locked to various trial events and split by GO vs NOGO

import my, my.dataload
import numpy as np, pandas, kkpandas
from ns5_process import LBPB




# Put this in dataload
class SpikeServer:
    @classmethod
    def get(self, **kwargs):
        return my.dataload.ulabel2spikes(**kwargs)

# Put this default and other defaults in dataload
tpk = {
    'labels': LBPB.mixed_stimnames,
    'label_kwargs': [{'stim_name':s} for s in LBPB.mixed_stimnames],
    'nonrandom' : 0,
    'outcome' : 'hit'
    }




gets = my.dataload.getstarted()

# Which units to analyze
units_to_analyze = gets['unit_db'][gets['unit_db'].include].index

for ulabel in units_to_analyze:
    my.printnow(ulabel)
    trials_info = my.dataload.ulabel2trials_info(ulabel)
    time_picker = kkpandas.timepickers.TrialsInfoTimePicker(trials_info)
    
    dfolded = kkpandas.pipeline.pipeline(trials_info,
        spike_server=SpikeServer,
        spike_server_kwargs={'ulabel': ulabel, 'sort_spikes': True},
        time_picker=time_picker,
        time_picker_kwargs={'event_name': 'stim_onset'},
        trial_picker_kwargs=tpk,
        folding_kwargs={'dstart': -2.0, 'dstop': 2.0},
        )
    
    dfolded2 = my.dataload.ulabel2dfolded(ulabel,
        trial_picker_kwargs=tpk,
        folding_kwargs={'dstart': -2.0, 'dstop': 2.0},
        )
    
    # Test the same
    for label in dfolded:
        folded1 = dfolded[label]
        folded2 = dfolded2[label]
        assert kkpandas.is_equal(folded1, folded2)
        assert kkpandas.is_equal(folded2, folded1)
    