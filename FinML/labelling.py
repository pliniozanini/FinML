import pandas as pd
import numpy as np

from .filters import getTEvents


class Labeller:
    def __init__(
        self,
        target_series: pd.Series,
    ) -> None:
        self.series = target_series

    @staticmethod
    def apply_tripple_barrier(
        close: pd.Series,
        events: pd.DataFrame,
        pt_sl,
        molecule: np.ndarray = None
    ) -> pd.DataFrame:
        '''
        Labeling observations using tripple-barrier method
        
            Parameters:
                close (pd.Series): close prices of bars
                events (pd.DataFrame): dataframe with columns:
                                    - t1: The timestamp of vertical barrier (if np.nan, there will not be
                                            a vertical barrier)
                                    - trgt: The unit width of the horizontal barriers
                pt_sl (list): list of two non-negative float values:
                            - pt_sl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                                        If 0, there will not be an upper barrier.
                            - pt_sl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                                        If 0, there will not be a lower barrier.
                molecule (np.ndarray):  subset of event indices that will be processed by a
                                        single thread (will be used later)
            
            Returns:
                out (pd.DataFrame): dataframe with columns [pt, sl, t1] corresponding to timestamps at which
                                    each barrier was touched (if it happened)
        '''
        if molecule is not None:
            events_ = events.loc[molecule]
        else:
            events_ = events
        out = events_[['t1']].copy(deep=True)
        if pt_sl[0] > 0:
            pt = pt_sl[0] * events_['trgt']
        else:
            pt = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
        if pt_sl[1] > 0:
            sl = -pt_sl[1] * events_['trgt']
        else:
            sl = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
        
        for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
            df0 = close[loc: t1]                                       # path prices
            df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']     # path returns
            out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()        # earlisest stop loss
            out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()        # earlisest profit taking
        return out

    @staticmethod
    def get_events_tripple_barrier(
        close: pd.Series, tEvents: np.ndarray, pt_sl: float, trgt: pd.Series, minRet: float,
        numThreads: int = 1, t1 = False, side: pd.Series = None
    ) -> pd.DataFrame:
        '''
        Getting times of the first barrier touch
        
            Parameters:
                close (pd.Series): close prices of bars
                tEvents (np.ndarray): np.ndarray of timestamps that seed every barrier (they can be generated
                                    by CUSUM filter for example)
                pt_sl (float): non-negative float that sets the width of the two barriers (if 0 then no barrier)
                trgt (pd.Series): s series of targets expressed in terms of absolute returns
                minRet (float): minimum target return required for running a triple barrier search
                numThreads (int): number of threads to use concurrently
                t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                                to disable vertical barriers)
                side (pd.Series) (optional): metalabels containing sides of bets
            
            Returns:
                events (pd.DataFrame): dataframe with columns:
                                        - t1: timestamp of the first barrier touch
                                        - trgt: target that was used to generate the horizontal barriers
                                        - side (optional): side of bets
        '''
        trgt = trgt.loc[trgt.index.intersection(tEvents)]
        trgt = trgt[trgt > minRet]
        if t1 is False:
            t1 = pd.Series(pd.NaT, index=tEvents)
        if side is None:
            side_, pt_sl_ = pd.Series(np.array([1.] * len(trgt.index)), index=trgt.index), [pt_sl[0], pt_sl[0]]
        else:
            side_, pt_sl_ = side.loc[trgt.index.intersection(side.index)], pt_sl[:2]
        events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
        df0 = apply_tripple_barrier(close, events, pt_sl_, events.index)

        events['t1'] = df0.apply(lambda x: x.dropna().min(), axis=1) #df0.dropna(how='all').min(axis=1)
        if side is None:
            events = events.drop('side', axis=1)
        return events

    @staticmethod
    def add_vertical_barrier(close: pd.Series, tEvents: np.ndarray, numDays: int) -> pd.Series:
        t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
        t1 = t1[t1 < close.shape[0]]
        t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])    # adding NaNs to the end
        return t1
    
    @staticmethod
    def get_bins(close: pd.Series, events: pd.DataFrame, t1=False) -> pd.DataFrame:
        
    @staticmethod
    def get_vol(close: pd.Series, span0: int = 20) -> pd.Series:
        df0 = close / close.shift(1) - 1    # returns
        df0 = df0.ewm(span=span0).std()
        return df0
    
    def label_binarizer(self, events: pd.DataFrame) -> pd.DataFrame:
        if events is None:
            events = getTEvents(
                self.series,
                self.series.std(),
                lambda x: 2*x.std()
            )
        t1 = self.add_vertical_barrier(self.series, events, numDays=7)
        events_ = self.get_events_tripple_barrier(
            close=self.series, tEvents=events, pt_sl=[1, 1],
            trgt=self.get_daily_vol(self.series), minRet=0.000001,
            numThreads=1, t1=t1
        )
        labels = self.get_bins(close=ibov['Close'], events=events_, t1=t1)
        return labels

