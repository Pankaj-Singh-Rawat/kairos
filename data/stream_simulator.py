'''
This script will load the PaySim CSV, sort it by the step column (which represents hours), and yield batches. At step == 150, it will inject covariate shift by multiplying the transaction amount by 3 for legitimate transfers.
'''

import pandas as pd
import numpy as np

class DataStreamer:
    def __init__(self, file_path: str, drift_start_step: int = 150):
        # load data and sort by time step
        self.df = pd.read_csv(file_path).sort_values(by='step').reset_index(drop=True)
        self.drift_start_step = drift_start_step
        self.unique_step = self.df['step'].unique()
        self.current_index = 0


    def get_next_batch(self):
        if self.current_index >= len(self.unique_step):
            return None # stram is finished in this scenerio

        step = self.unique_step[self.current_index]
        batch = self.df[self.df['step']==step].copy()

        if step >= self.drift_start_step:
            batch = self._inject_drift(batch)

        self.current_index += 1
        return batch

    def _inject_drift(selfself, batch: pd.DataFrame) -> pd.DataFrame:
        # Covariate shift: Fraudsters change tactics, or holiday shopping spikes transaction amounts
        # Inflating 'amount' specifically for TRANSFER types to create a clean statistical anomaly
        mask = (batch['type'] == 'TRANSFER') & (batch['isFraud'] == 0)
        batch.loc[mask, 'amount'] = batch.loc[mask, 'amount'] * 3.5
        return batch