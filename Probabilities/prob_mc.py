import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/hans/Documents/research/HI_galaxy_search/analytical")
from cpp_amf_wrapper import correlation_coefficient, amf
from util import fitted_peak_3x3, vec2ang, ang2vec

def produce_R (us, chord_theta, wavelength, m1, m2, delta_tau, time_samples):
    #first u should be the source
    R = np.identity(us.shape[0])
    for i in range(us.shape[0]-1):
        for j in range(i+1, us.shape[0]):
            cc = correlation_coefficient (us[i], us[j], chord_theta, wavelength, m1, m2, delta_tau, time_samples)
            R[i][j] = cc
            R[j][i] = cc
    return R

def produce_R_with_quadratic_interp (us, chord_theta, wavelength, m1, m2, delta_tau, time_samples):
    #first u should be the source
    
    sdt = wavelength/(m1*8.5)/8 #small delta thetas
    sdp = wavelength/(m2*6.3)/8 #small delta phi
    
    R = np.identity(us.shape[0])
    for i in range(us.shape[0]-1):
        for j in range(i+1, us.shape[0]):
        		source_theta, source_phi_0 = vec2ang(us[i])
        		dest_theta, dest_phi = vec2ang(us[j])
        		
        		around_dest = np.empty([9,3])
        		around_dest[0] = ang2vec(dest_theta-sdt, dest_phi - sdp)
        		around_dest[1] = ang2vec(dest_theta-sdt, dest_phi)
        		around_dest[2] = ang2vec(dest_theta-sdt, dest_phi + sdp)
        		around_dest[3] = ang2vec(dest_theta, dest_phi - sdp)
        		around_dest[4] = ang2vec(dest_theta, dest_phi)
        		around_dest[5] = ang2vec(dest_theta, dest_phi + sdp)
        		around_dest[6] = ang2vec(dest_theta+sdt, dest_phi - sdp)
        		around_dest[7] = ang2vec(dest_theta+sdt, dest_phi)
        		around_dest[8] = ang2vec(dest_theta+sdt, dest_phi + sdp)
        		
        		ccm = amf (chord_theta, wavelength, source_theta, source_phi_0, m1, m2, around_dest, delta_tau, time_samples).reshape([3,3])
        		
        		peak, peakx, peaky = fitted_peak_3x3(ccm, 1, 1)
        		
            R[i][j] = peak
            R[j][i] = peak
    return R

def montecarlo_probability (R, nsigma_source, nsigma_threshold, max_attempts=10, tol=0.001, seed=1234):
    samples_per_attempt=1000000
    mu = R[0] * nsigma_source
    cov = R
    
    #old wrong definitions
    #mu = np.insert(R[:num_aliases],0,1) #partial definition of mu to help the definition of cov. Will be modified later.
    #cov = np.tile(mu,(mu.shape[0],1)) * np.tile(mu,(mu.shape[0],1)).T
    #np.fill_diagonal(cov,1)
    #mu = mu * nsigma_source
    
    rng = np.random.default_rng(seed=seed)
    
    nsamples=0
    detectSum=0
    confusionSum=0
    noConfusionProbability=-1
    for i in range(max_attempts):
        samples = rng.multivariate_normal(mu, cov, size=samples_per_attempt)
        nsamples += samples_per_attempt
        detects = np.any(samples>nsigma_threshold,axis=1)
        detectSum += np.sum(detects)
        samples_with_detects = samples[np.nonzero(detects)]
        confusionSum += np.sum(np.any(samples_with_detects[:,1:] > samples_with_detects[:,0][np.newaxis].T,axis=1))
        temp_ncp = 1-(detectSum+confusionSum)/nsamples
        if np.abs(noConfusionProbability-temp_ncp) < tol:
            noConfusionProbability = temp_ncp
            break
        noConfusionProbability = temp_ncp
    return 1-detectSum/nsamples, confusionSum/nsamples

if __name__ == "__main__":
    rvals = np.linspace(0,0.99)
    probs = np.empty([rvals.shape[0],2])
    for i in range(rvals.shape[0]):
        probs[i] = montecarlo_probability(np.array([rvals[i], rvals[i]]),6,5)
        print("\x1b[2K",str((i+1)/rvals.shape[0] * 100)+"% complete", end='\r')
    print("\n")
    plt.plot(rvals,probs[:,0],label="no detection")
    plt.plot(rvals,probs[:,1]+probs[:,0],label="confusion or no detection")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Probability")
    plt.title("Alias Confusion")
    plt.legend()
    plt.show()
