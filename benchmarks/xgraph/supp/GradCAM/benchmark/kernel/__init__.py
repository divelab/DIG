import os
from definitions import ROOT_DIR
import sys
sys.path.append(os.path.abspath(os.path.join(ROOT_DIR, '..', 'metrics')))
print(f"Add {os.path.abspath(os.path.join(ROOT_DIR, '..', 'metrics'))} as a system path.")
from .evaluation import test, acc_score
from .initial import init
from .explain import XCollector, sample_explain
