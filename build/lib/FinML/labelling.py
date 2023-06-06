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
        """
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
        """  # noqa: E501
        if molecule is not None:
            events_ = events.loc[molecule]
        else:
            events_ = events
        out = events_[['t1']].copy(deep=True)
        if pt_sl[0] > 0:
            pt = pt_sl[0] * events_['trgt']
        else:
            pt = pd.Series(data=[np.nan] * len(events.index), index=events.index)
        if pt_sl[1] > 0:
            sl = -pt_sl[1] * events_['trgt']
        else:
            sl = pd.Series(data=[np.nan] * len(events.index), index=events.index)
        
        for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
            df0 = close[loc: t1]                                       # path prices
            df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']     # path returns
            out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()        # earlisest stop loss
            out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()        # earlisest profit taking
        return out

    @staticmethod
    def get_events_tripple_barrier(
        close: pd.Series, tEvents: np.ndarray, pt_sl: float, trgt: pd.Series,
        minRet: float = 0,
        numThreads: int = 1,
        t1=False,
        side: pd.Series = None
    ) -> pd.DataFrame:
        """
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
        """  # noqa E501
        trgt = trgt.loc[trgt.index.intersection(tEvents)]
        trgt = trgt[trgt > minRet]
        if t1 is False:
            t1 = pd.Series(pd.NaT, index=tEvents)
        if side is None:
            side_, pt_sl_ = pd.Series(np.array([1.] * len(trgt.index)), index=trgt.index), [pt_sl[0], pt_sl[0]]
        else:
            side_, pt_sl_ = side.loc[trgt.index.intersection(side.index)], pt_sl[:2]
        events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
        df0 = Labeller.apply_tripple_barrier(close, events, pt_sl_, events.index)

        events['t1'] = df0.apply(lambda x: x.dropna().min(), axis=1)
        if side is None:
            events = events.drop('side', axis=1)
        return events

    @staticmethod
    def add_vertical_barrier(
        close: pd.Series, tEvents: np.ndarray, numDays: int
    ) -> pd.Series:
        t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
        t1 = t1[t1 < close.shape[0]]
        return pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])    # adding NaNs to the end
    
    @staticmethod
    def get_bins(close: pd.Series, events: pd.DataFrame, t1=False) -> pd.DataFrame:
        """
        Generating labels with possibility of knowing the side (metalabeling)
        
            Parameters:
                close (pd.Series): close prices of bars
                events (pd.DataFrame): dataframe returned by 'get_events' with columns:
                                    - index: event starttime
                                    - t1: event endtime
                                    - trgt: event target
                                    - side (optional): position side
                t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                                to disable vertical barriers)

            Returns:
                out (pd.DataFrame): dataframe with columns:
                                        - ret: return realized at the time of the first touched barrier
                                        - bin: if metalabeling ('side' in events), then {0, 1} (take the bet or pass)
                                                if no metalabeling, then {-1, 1} (buy or sell)
        """  # noqa E501
        events_ = events.dropna(subset=['t1'])
        px = events_.index.union(
            (
                pd.DatetimeIndex(events_['t1'].values)
                .tz_localize(events_.index.tz)
            )
        ).drop_duplicates()

        px = close.reindex(px, method='bfill')
        out = pd.DataFrame(index=events_.index)
        out['ret'] = px.loc[
            pd.DatetimeIndex(
                events_['t1'].values
            ).tz_localize(px.index.tz)
        ].values / px.loc[events_.index] - 1
        if 'side' in events_:
            out['ret'] *= events_['side']
        out['bin'] = np.sign(out['ret'])
        if 'side' in events_:
            out.loc[out['ret'] <= 0, 'bin'] = 0
        else:
            if t1 is not None:
                vertical_first_touch_idx = events_[
                    events_['t1'].isin(t1.values)
                ].index
                out.loc[vertical_first_touch_idx, 'bin'] = 0
        return out
        
    @staticmethod
    def get_vol(close: pd.Series, span0: int = 20) -> pd.Series:
        df0 = close / close.shift(1) - 1    # returns
        df0 = df0.ewm(span=span0).std()
        return df0
    
    def label_binarizer(self, events: pd.DataFrame = None) -> pd.DataFrame:
        if events is None:
            events = self.series.index
        t1 = self.add_vertical_barrier(self.series, events, numDays=7)
        events_ = self.get_events_tripple_barrier(
            close=self.series, tEvents=events, pt_sl=[1, 1],
            trgt=self.get_vol(self.series), minRet=0,
            numThreads=1, t1=t1
        )
        labels = self.get_bins(close=self.series, events=events_, t1=t1)
        return labels
