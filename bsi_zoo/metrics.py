from sklearn.metrics import jaccard_score

def jaccard_error(x, x_hat):
    # You can read more on the Jaccard score in Scikit-learn definition https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
    return 1 - jaccard_score(x, x_hat, average='samples')
 
def dummy(x, x_hat):
    return 123

def dummy2(x, x_hat):
    return 1234


