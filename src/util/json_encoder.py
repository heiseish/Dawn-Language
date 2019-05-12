''' Convert object to json dumpable object'''
from typing import Union

import json
import numpy


def encode(data: any) -> str:
    ''' Serialize object of different numpy types into json string
    Args:
        data (Union[numpy.integer, numpy.floating, numpy.ndarray]): object to be encoded
    '''

    class MyEncoder(json.JSONEncoder):
        ''' Extends intance of json.JSONENcoder
        Args:
            object to be encoded
        Returns:
            encoded string
        '''

        def default(self,
                    obj: Union[numpy.integer, numpy.floating, numpy.ndarray]):
            if isinstance(obj, numpy.integer):
                return int(obj)
            if isinstance(obj, numpy.floating):
                return float(obj)
            if isinstance(obj, numpy.ndarray):
                return obj.tolist()
            return super(MyEncoder, self).default(obj)

    return json.dumps(data, cls=MyEncoder)
