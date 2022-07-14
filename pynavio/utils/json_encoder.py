import json

import pandas as pd


class JSONEncoder(json.JSONEncoder):

    def default(self, obj: object):
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)
