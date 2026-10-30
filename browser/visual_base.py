import numpy as np
import pandas as pd

def pd_one_epoch_to_csv(args, environment, export_data, data_type, epoch, data_per_epoch, data_per_subsystem, PATH):
    for i in range(data_per_epoch):
        export_data_temp = []
        for (j, subsystem) in enumerate(environment.subsystems_list):
            export_data_temp.append(eval('subsystem.' + data_type + '_list')[epoch * data_per_epoch + i])
        export_data_line = np.array(export_data_temp, dtype=object)
        export_data.append(export_data_line.reshape(-1))
    data = np.array(export_data)
    data = pd.DataFrame(data=data)
    data.to_csv(PATH, index=True)
    return export_data
