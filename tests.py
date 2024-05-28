import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

# discrepancies are considered right-sided by default
# hence tests based on the p-value have a minus sign in front 
# because more discrepant results return smaller values

def min_p(ref,data,rng=None):
    # ref: nxd numpy array
    # data: mxd numpy array
    
    p_ref, p_data = return_pvalues(ref,data,rng=rng)
    return -np.log(np.min(p_ref,axis=1)), -np.log(np.min(p_data,axis=1))

def fused_p(ref,data,T=1,rng=None):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array
    
    p_ref, p_data = return_pvalues(ref,data,rng=rng)
    return -np.log(fusion(p_ref,-T)), -np.log(fusion(p_data,-T))


def avg_p(ref,data,rng=None):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array

    p_ref, p_data = return_pvalues(ref,data,rng=rng)
    return -np.log(np.mean(p_ref,axis=1)), -np.log(np.mean(p_data,axis=1))

def fused_t(ref,data,T):
    fused_ref = fusion(ref,T)
    fused_data = fusion(data,T)

    return emp_pvalues(fused_ref,fused_data)

def emp_pvalue(ref,t):
    # this definition is such that p=1/(N+1)!=0 if data is the most extreme value
    p = (np.count_nonzero(ref >= t)+1) / (len(ref)+1)
    return p

def emp_pvalues(ref,data):
    return np.array([emp_pvalue(ref,t) for t in data])

def p_to_z(pvals):
    return norm.ppf(1 - pvals)

def z_to_p(z):
    # sf=1-cdf (sometimes more accurate than cdf)
    return norm.sf(z)

def Zscore(ref,data):
    return p_to_z(emp_pvalues(ref,data))

def power(t_ref,t_data,zalpha=[.5,1,2,2.5]):
    # alpha=np.array([0.309,0.159,0.06681,0.0228,0.00620]))
    alpha = z_to_p(zalpha)
    quantiles = np.quantile(t_ref,1-alpha)
    return p_to_z(alpha), emp_pvalues(t_data,quantiles)

def fusion(x,T):
    return T * logsumexp(1/T*x, axis=1, b=1/x.shape[1])

def bootstrap_pn(pn,rng=None):
    # if rng=None, check_rng(rng) returns a rnd numb generator with random seed
    # if rng=int, check_rng(rng) returns a rnd numb generator with seed=int
    # if rng=np.rando.default_rng(seed), check_rng(rng) returns the rnd number generator with the given seed
    
    rnd = check_rng(rng)

    return rnd.choice(pn,size=len(pn))

def bootstrap_pval(pn,t,rng=None):

    return emp_pvalue(bootstrap_pn(pn,rng=rng),t)


def return_pvalues(ref,data,rng=None):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value by bootstrapping col (the value of t is removed first)
        p_ref[:,idx] = np.transpose([bootstrap_pval(np.delete(col,idx2),el,rng=rng) for idx2, el in enumerate(col)])

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data

def check_rng(rnd):
    if rnd is None:
        return np.random.default_rng(seed=rnd)
    elif isinstance(rnd, int):
        return np.random.default_rng(seed=rnd)
    else:
        return rnd
    #    if rnd is None:
    #        return np.random.default_rng(seed=rnd)
    #    elif isinstance(rnd, int):
    #        return np.random.default_rng(seed=rnd)
    #    elif isinstance(rnd, np.random.default_rng):
    #        return rnd
    #    raise ValueError(
    #        "%r rnd must be None, int or and istance of np.random.default_rng" % rnd
    #    )