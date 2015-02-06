import htkdb, os, time
import argparse, logging, sys
from mercurial import ui, hg, localrepo

parser = argparse.ArgumentParser()
parser.add_argument('--skip_repo', dest='skip_repo', action='store_true',
                        default=False, help='Do not check repository state')
parser.add_argument('db_name', help='Name of training database')
parser.add_argument('db_path', help='Path to database')
parser.add_argument('--utt2spk', dest='utt2spk',  action='store', 
                     type=str, default=None,
                     help='Location of utterances to speaker file')

arguments = parser.parse_args()
if not arguments.skip_repo:
    rep = localrepo.instance(ui.ui(), '.', False)
    if sum([len(x) for x in rep.status()]) != 0:
        print "Please commit changes to repository before running program"
        sys.exit(1)

logPath = os.path.join(arguments.db_path, arguments.db_name, "log.txt")
if os.path.exists(logPath): os.remove(logPath)
logging.basicConfig(filename=logPath, level=logging.INFO)
logging.info(" ".join(sys.argv))
logging.info(time.time())
rep = localrepo.instance(ui.ui(), '.', False)
revision_num = rep.changelog.headrevs()[0] 

logging.info("Revision number for code: " + str(revision_num))
db=htkdb.htkdb(arguments.db_name, arguments.db_path)
db.create_db(arguments.utt2spk)
