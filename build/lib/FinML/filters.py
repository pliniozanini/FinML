import pandas as pd
import numpy as np

# symmetrical CUSUM filter
def getTEvents(gRaw: pd.Series, h: float, floating_h_func=None, lags=20) -> np.ndarray:
    gRaw = gRaw[~gRaw.index.duplicated(keep='first')]
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if floating_h_func:
            h = floating_h_func(gRaw.loc[:i].iloc[-lags:])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)