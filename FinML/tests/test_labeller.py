import pandas as pd
from FinML.labelling import Labeller


def test_label_binarizer():
    # Create sample data
    y = pd.Series(
        [1, 2, 3, 1.5],
        index=list(pd.date_range('2020-01-01', periods=4, freq='D'))
    )

    # Test binarizing a single label
    lb = Labeller(y)
    output = lb.label_binarizer()
    print(output.to_dict())
    expected_out = pd.DataFrame(
        {
            'ret': [-0.5],
            'bin': [-1.0]
        },
        index=pd.DatetimeIndex(['2020-01-03'])
    )
    pd.testing.assert_frame_equal(output, expected_out)
