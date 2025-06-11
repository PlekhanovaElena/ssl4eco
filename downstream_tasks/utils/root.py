import os

# This can be messed up if multiple things are run at the same time
# But this variable is only accessed in the beginning when everything is
# instantiated (except for loading special test-datasets)
# But given that is environment-specific, it should not change between
# runs anyway
def init_root(path=None):
    if path is None:
        path = "/shares/wegner.ics.uzh/eplekh"
    if os.environ.get("SECOECO_ROOT") is None:
        os.environ["SECOECO_ROOT"] = path


def get_root():
    init_root()
    return os.environ["SECOECO_ROOT"]
