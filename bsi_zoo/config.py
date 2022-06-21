import os


def get_leadfield_path(subject, type=""):
    username = os.environ.get("USER")
    if subject is not None:
        if "anuja" in username or "hashemi" in username:
            if type == "free":
                return "bsi_zoo/tests/data/lead_field_free_%s.npz" % subject
            elif type == "fixed":
                return "bsi_zoo/tests/data/lead_field_%s.npz" % subject
    else:
        return 1


def get_fwd_fname(subject):
    username = os.environ.get("USER")
    if "anuja" in username or "hashemi" in username:
        return "bsi_zoo/tests/data/%s-fwd.fif" % subject
