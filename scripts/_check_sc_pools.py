import sys, json
sys.path.insert(0,'.')
from src.data.splits import load_manifest
m = load_manifest('Sc', 0)
print('All pool keys:', list(m['pools'].keys()))
