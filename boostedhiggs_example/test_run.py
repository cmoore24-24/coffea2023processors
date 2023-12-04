import warnings
warnings.filterwarnings('ignore', 'binning')
warnings.filterwarnings('ignore', 'JERSF')
warnings.filterwarnings('ignore','Port')
warnings.filterwarnings('ignore', "The 'nopython'")
warnings.filterwarnings('ignore', "Pandas")
warnings.filterwarnings('ignore', "Schedd")
warnings.filterwarnings('ignore')
from boostedhiggs.hbbprocessor_test import HbbProcessor
from boostedhiggs import corrections
from coffea.nanoevents import NanoEventsFactory, BaseSchema, PFNanoAODSchema
import json
import dask
#from distributed import Client
#from lpcjobqueue import LPCCondorCluster
#cluster = LPCCondorCluster(
#  ship_env=True,
#  log_directory='/uscmst1b_scratch/lpc1/3DayLifetime/cjmoore/mylog',
#  #memory='4718592000'
#)
#cluster.adapt(minimum=0, maximum=10)
#client = Client(cluster)
with open("jsons/qcd_and_more_hj_files.json") as fin:
    filelist = json.load(fin)
events = NanoEventsFactory.from_root(
    {filelist['Hbb'][0]:"Events"},
    permit_dask=True,
    metadata={"dataset": "Hbb"},
    #chunks_per_file=3000,
).events()
result = HbbProcessor().process(events)
dask.compute(result, scheduler='synchronous')
