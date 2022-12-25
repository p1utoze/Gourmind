import os
from hub import hub_handler
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import base64


MODEL_DIR = os.getenv("MODEL_DIR")
agri_df = pd.read_csv(os.path.join(MODEL_DIR, "ICRISAT-District Level Data.csv"))

agri_df.drop(agri_df.iloc[:, -6:], inplace=True, axis=1)
agri_df.rename(columns = {'State Code':'state_code', 'State Name':'state',
                              'Dist Name':'dist'}, inplace = True)

d= {}
for i in agri_df.columns:
    d[i]=("_".join(i.split()[:2])).lower()
    
agri_df.rename(columns = d, inplace = True)

ka_df = agri_df[agri_df["state"]=="Karnataka"]
ka_df = ka_df.replace(to_replace=["Bijapur / Vijayapura", "Gulbarga / Kalaburagi", "Kodagu / Coorg"], value=["Vijaypura", "Gulbarga", "Coorg"])

class Crop(object):
    def __init__(self, dist, crop):
        self.dist = dist
        self.crop = crop
        self.dist_data = ka_df[ka_df["dist"] == self.dist]
        self.pp_crop = [x for x in self.dist_data.iloc[:, 5:].columns if x.startswith(self.crop)]
        
    def plotter(self):
        fig, axe = plt.subplots(3, figsize=(10, 10))
        sns.lineplot(x="year", y=self.pp_crop[0], data=self.dist_data, color="r", ax=axe[0])
        sns.lineplot(x="year", y=self.pp_crop[1], data=self.dist_data, color="g", ax=axe[1])
        sns.lineplot(x="year", y=self.pp_crop[2], data=self.dist_data, color="b", ax=axe[2])
        fig.tight_layout(pad = 1.5)
        plt.savefig('/tmp/plot.png')
        
    def pp_model(self):
        m = Prophet()
        df = self.dist_data.loc[:, ["year", self.pp_crop[-1]]]
        cols = df.columns
        df.rename(columns={cols[0]: "ds", cols[1]: "y"}, inplace=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        fig1 = m.plot(forecast)
        


@hub_handler
def inference_handler(inputs, _):
    '''The main inference function which gets triggered when the API is invoked'''
    
    dataplot = Crop(inputs['district'].strip().capitalize(), inputs['crop_name'].strip().lower())
    dataplot.plotter()
    
    with open('/tmp/plot.png', 'rb') as f:
        img_bytes = f.read()

        # convert to a base64 string
        output = base64.b64encode(img_bytes).decode('utf-8')    
    
    return output
