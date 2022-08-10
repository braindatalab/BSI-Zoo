def get_leadfield_path(subject, type=""):
    if subject is not None:
        if type == "free":
            return "bsi_zoo/tests/data/lead_field_free_%s.npz" % subject
        elif type == "fixed":
            return "bsi_zoo/tests/data/lead_field_%s.npz" % subject
    else:
        return None


def get_fwd_fname(subject):
    return "bsi_zoo/tests/data/%s-fwd.fif" % subject
