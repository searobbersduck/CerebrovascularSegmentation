import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(root)
sys.path.append(root)

from external_lib.MedCommon.utils.datasets_utils import DatasetsUtils


