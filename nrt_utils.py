# Copyright (C) 2024 European Union (Joint Research Centre)
# author(s): Kenji OSE

from nrt.monitor.ewma import EWMA
from nrt.monitor.iqr import IQR
from nrt.monitor.ccdc import CCDC
from nrt.monitor.cusum import CuSum
from nrt.monitor.mosum import MoSum

import os
import time
import itertools
from multiprocessing import Pool

import pandas as pd
import numpy as np
import xarray as xr


class Nrt_launch():
    """Launch NRT method in one way

    Args:
        method (str): name of the NRT method
        trend (bool): take into account trend (default is False)
        harmonic_order (int): harmonic order of the fitting model (default is 2)
        mask (int): array non-forest mask (default is None)
        **kwargs: monitoring parameters

    Returns:
        Nrt object
    """

    def __init__(self, method, trend=False, harmonic_order=2,
                 mask=None, **kwargs):

        self.method = method
        self.trend = trend
        self.harmonic_order = harmonic_order
        self.mask = mask
        self.kwargs = kwargs
        self._calls = {'EWMA': self._callEwma,
                       'IQR': self._callIqr,
                       'CCDC': self._callCcdc,
                       'CUSUM': self._callCusum,
                       'MOSUM': self._callMosum
                      }

        self.call = self._calls.get(method, self._callErr)

    def _callEwma(self):
        nrt = EWMA(trend=self.trend, harmonic_order=self.harmonic_order,
                   mask=self.mask, **self.kwargs)
        return nrt

    def _callIqr(self):
        nrt = IQR(trend=self.trend, harmonic_order=self.harmonic_order,
                  mask=self.mask, **self.kwargs)
        return nrt

    def _callCcdc(self):
        nrt = CCDC(trend=self.trend, harmonic_order=self.harmonic_order,
                   mask=self.mask, **self.kwargs)
        return nrt

    def _callCusum(self):
        nrt = CuSum(trend=self.trend, harmonic_order=self.harmonic_order,
                    mask=self.mask, **self.kwargs)
        return nrt

    def _callMosum(self):
        nrt = MoSum(trend=self.trend, harmonic_order=self.harmonic_order,
                    mask=self.mask, **self.kwargs)
        return nrt

    def _callErr(self):
        raise ValueError


class Nrt_run():
    """Create Nrt instance

    Args:
        ar_index (array): time series array for one XY position
        method (str): name of the NRT method
        date_pivot (datetime): pivot date between fitting and monitoring periods
        gdf (geoDataFrame): geoDataFrame of XY points (default is None)
        gdf_field (str): date field of gdf [in Unix time] (default is None)
    """

    def __init__(self, ar_index, method, date_pivot, gdf=None, gdf_field=None):
        self.ar_index = ar_index
        self.method = method
        self.gdf = gdf
        self.gdf_field = gdf_field 
        self.date_pivot = date_pivot
        if gdf is not None and gdf_field is not None:
            self.nrt_ref()

    def nrt_run(self, params, outfile, mask=None):
        """Run NRT methods

        Args:
            params (dict): NRT method parameters
            outfile (str): output filepath
            mask (array): raster mask (default is None)
        """
        # input array
        arr_hist = self.ar_index.sel(time=slice(None, self.date_pivot))
        arr_moni = self.ar_index.sel(time=slice(self.date_pivot, None))
        # launch method
        test = Nrt_launch(self.method, mask=mask, **params).call()
        # fitting
        test.fit(dataarray=arr_hist)
        # monitoring
        for array, date in zip(arr_moni.values, 
                               arr_moni.time.values.astype('datetime64[s]').tolist()):
            test.monitor(array=array, date=date)

        test.report(outfile, layers=['mask', 'detection_date'])

        with open(f"{os.path.splitext(outfile)[0]}_params.txt", 'w') as pfile:
            pfile.write(f"Method: {self.method}\n---\n")
            pfile.write("Parameters:\n")
            pfile.write(f"{params}")

    def find_nearest_indices(self, x, y):
        """Find nearest pixel index from input x and y

        Args:
            x (float): x coordinates
            y (float): y coordinates

        Return: tuple of nearest x, y index (int)
        """
        distances = np.sqrt((self.ar_ref.x - x)**2 + (self.ar_ref.y - y)**2)
        # Find the point index with the shortest distance
        nearest_index = np.unravel_index(distances.argmin(), distances.shape)
        return nearest_index

    def nrt_ref(self):
        """Convert input reference vector into array coherent, in terms of dims,
        with the time-series datacube

        """
        coord_list = {"x":[], "y":[]}
        for _, row in self.gdf.iterrows():
            coord_list["x"].append(row.geometry.x)
            coord_list["y"].append(row.geometry.y)

        x = list(set(coord_list["x"]))
        y = list(set(coord_list["y"]))
        self.ar_index_lite = self.ar_index.sel(x=x, y=y, method='nearest')
        self.ar_ref = xr.DataArray(np.nan, dims=('x', 'y'), 
                                   coords={'x': self.ar_index_lite['x'],
                                           'y': self.ar_index_lite['y']
                                          })

        for _, row in self.gdf.iterrows():
            x = row.geometry.x
            y = row.geometry.y
            nearest_indices = self.find_nearest_indices(x, y)
            self.ar_ref.loc[{'x': self.ar_ref['x'][nearest_indices[0]],
                             'y': self.ar_ref['y'][nearest_indices[1]]}] = row[self.gdf_field]

    def nrt_stat(self, params):
        """Run NRT methods and return time lag stats

        Args:
            params (list): list of dict() that contain method parameters

        Return: dictionnary of scores
        """
        # get reference coordinates
        arr_hist = self.ar_index_lite.sel(time=slice(None, self.date_pivot))
        arr_moni = self.ar_index_lite.sel(time=slice(self.date_pivot, None))

        test = Nrt_launch(self.method, **params).call()

        # fitting
        test.fit(dataarray=arr_hist)
        # monitoring
        for array, date in zip(arr_moni.values,
                               arr_moni.time.values.astype('datetime64[s]').tolist()):
            test.monitor(array=array, date=date)

        self.detect_date = test.detection_date.T
        self.detect_date = self.detect_date.astype(float)
        self.detect_date[self.detect_date == 0] = np.nan

        self.diff_date = np.abs(self.detect_date - self.ar_ref)
        self.diff_date = self.diff_date.values.ravel()
        self.diff_date = self.diff_date[~np.isnan(self.diff_date)]

        out_dict = dict()
        if len(self.diff_date) > 0:
            s = pd.Series(self.diff_date)
            out_dict.update(params)
            stats = s.describe()
            for stat in stats.axes[0]: 
                if stat == 'count':
                    out_dict["tp"] = round(stats[stat]/len(self.gdf), 2)
                out_dict[stat] = round(stats[stat], 2)

        return out_dict

    def nrt_stat_in_parallel(self, params, nb_proc=4):
        """nrt_stat in multiprocessing mode

        Args:
            params (list): list of dict() that contain method parameters
            nb_proc (int): number of processors (default is 4)

        Return: dataframe of scores
        """
        time_start = time.time()
        pool = Pool(processes=nb_proc)
        out_res = pool.map(self.nrt_stat, params)
        pool.close()
        pool.join()
        out_df = pd.DataFrame.from_records(out_res)
        duration = (time.time() - time_start)/60
        print(f'duration: {duration} min')

        return out_df


def params_bounds(params):
    """Make all possible parameters combinations

    Args:
        params (list): list of dict() that contain method parameters
                ex. [{'param1': (min, max, step),
                      'param2': [val01, val02, val03]},
                       ...]

    Return: list of dictionnaries
    """

    p_names = list()
    p_values = list()
    p_len = len(params)

    for i in params:
        p_names.append(i)
        if isinstance(params[i], tuple):
            p = list(np.arange(params[i][0], params[i][1], params[i][2]))
        if isinstance(params[i], list):
            p = params[i]
        p_values.append(p)

    combin = itertools.product(*p_values)

    out_list = list()
    for c in combin:
        out_dico = dict()
        for i in range(p_len):
            out_dico[p_names[i]] = c[i]
        out_list.append(out_dico)

    return out_list