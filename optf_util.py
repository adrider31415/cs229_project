import numpy as np
import cPickle as pickle 
import scipy.optimize
import matplotlib.pyplot as plt
import os

def chisq(vvec, svec, psd, A, t0):
    ws = 2.*np.pi*np.fft.rfftfreq(2*len(vvec)-1)
    intnum = vvec - A*np.exp(-1.j*ws*t0)*svec
    return np.einsum('i,i', np.conjugate(intnum), intnum/psd)



class Optf:
    #Class to load, filter, and hold pulse data.
    
    def __init__(self, fname):
        self.fname = fname #initialize object with filename
        self.datam = "Pulses not loaded"
        self.energies = "Pulse energies not loaded"
        self.iscollect = "Collection Bool not loaded"
        self.filts = "No filters loaded"
        self.npsd = "No noise PSD loaded"
        self.ofamps = "No filter amps"
        self.times = "chi sq min times not computed"#chi squared minimum time of each filter"
        self.gentimes = "No generated data computed"

    def load(self, findmin = 2):
        #Loads the fft data matrix and populates some atirbutes.
        dat = np.array(pickle.load(open(self.fname, 'rb')))
        self.datam = dat[:, 2:-2] #fft data matrix [pulse#, freq]
        self.energies = np.real(dat[:, -1]) #column of deposited energies
        self.iscollect = np.real(dat[:, -2]) #boole if charge collected.

    def load_filts(self, tempf, nfilts = 2, findmin = 2):
        #loads the filter template files.
        self.filts = []
        fid = open(tempf, 'rb')#open file
        for i in range(nfilts):
            temp = pickle.load(fid)[2:]#starting fft. 
            for f in self.filts:
                temp -= np.einsum('i,i', np.conjugate(f), temp)*f #Graham Schmid ortho
            temp /= np.sqrt(np.einsum('i,i', np.conjugate(temp), temp))#normalization 
            self.filts.append(temp)
        self.filts = np.array(self.filts)
        fid.close()

    def gen_data(self, npulse, window = 2048, s = 300., st = 10., sA = 1000.):
        #Generates fake Gaussian pulses for testing.
        #tarr = np.einsum('i,j-> ji', np.arange(window), np.ones(npulse))
        t0s = st*np.random.randn(npulse) + window/2. #random time offsets
        As = sA*np.abs(np.random.randn(npulse)) + 10.*sA #random amplitude
        tvec = np.arange(window) #time vector
        NFFT = int(window/2)+1 #number of fft points
        self.datam =  np.zeros((npulse, NFFT), dtype = 'complex_')
        for i, t0 in enumerate(t0s): #Loop over pulses generateing gaussian pulses
            g = lambda ti: (2.*np.pi*s**2)**(-1/2)*np.exp(-0.5*((ti-t0)/s)**2)
            self.datam[i] = As[i]*np.fft.rfft(g(tvec)) 

        self.gentimes = t0s #record true times
        self.energies = As #set energies to A
        self.filts = np.array([np.fft.rfft((2.*np.pi*s**2)**(-1/2)*np.exp(-0.5*((tvec-window/2.)/s)**2))]) #fft of g with 0 time offset
        self.psd = np.ones(NFFT) #uniform psd
        
    def fit(self):
        #minimizes chis^2 to find the optimal time offset between template and data then compute chi sq min amp estimate
        self.times = np.zeros((len(self.datam), len(self.filts)))#prealocate mem
        self.ofamps = np.zeros((len(self.datam), len(self.filts)))#preallocate mem
        NFFT = len(self.datam[0, :]) #Num points in fft of window
        ws = 2.*np.pi*np.fft.rfftfreq(2*NFFT-1) #ang freq vector
        for i, p in enumerate(self.datam):
            for j, s in enumerate(self.filts):
                f = lambda t: chisq(p, s, self.psd, 1., t)
                t0 = scipy.optimize.minimize_scalar(f, bounds = (0-1.*NFFT, 1.*NFFT), method = 'bounded', options = {'maxiter': 2000}).x
                T = np.exp(1.j*ws*t0)
                self.ofamps[i, j] = np.einsum('i,i', np.conjugate(s), T*p/self.psd)/np.einsum('i,i', np.conjugate(s), s/self.psd)
                self.times[i, j] = t0
        
    def pltgenfit(self, cent = 1024):
        #plots test pulses with bad time offset estimaters.
        v = plt.hist((cent + np.ravel(self.times)) - self.gentimes)
        plt.xlabel('time estimation error [samples]')
        plt.ylabel('Frequency per %i'%len(self.datam[:, 0]))
        plt.title("Time estimation error")
        plt.show()
        bins = np.linspace(-2, 2, 20)
        pers = 100.*(np.ravel(self.ofamps) - self.energies)/self.energies
        v2 = plt.hist(pers)
        plt.xlabel('% amplitude estimation error [%]')
        plt.ylabel('Frequency per %i'%len(self.datam[:, 0]))
        plt.title("Amplitude estimation error")
        plt.show()
        
        
    def saveOptf(self, outpath = '/home/arider/cs229_project/proc_data', clear_data = True):
        #pickles file after clearing self.datam.
        outf = os.path.join(outpath, os.path.split(self.fname)[-1])
        if clear_data:
            self.datam = 'Data matrix cleared'
        pickle.dump(self, open(outf, 'wb'))

    def loadOptf(self, outpath = '/home/arider/cs229_project/proc_data'):
        #Loads previously pickled files.
        outf = os.path.join(outpath, os.path.split(self.fname)[-1])
        tempobj = pickle.load(open(outf, 'rb'))
        self.datam = tempobj.datam
        self.energies = tempobj.energies
        self.iscollect = tempobj.iscollect
        self.filts = tempobj.filts
        self.npsd = tempobj.npsd
        self.ofamps = tempobj.ofamps
        self.times = tempobj.times
        self.gentimes = tempobj.gentimes

    def plt_pulses(self, cuts, skip = 10):
        #Plots pulses in time domain that satisfy all cuts. cuts is an array of boolean arrays
        bt = cuts[0]
        for b in cuts[1:]:
            bt = np.logical_and(bt, b)
        
        for i, p in enumerate(self.datam[bt, :]):
            if i%skip == 0:
                plt.plot(np.fft.irfft(p))

        plt.show()


def proc_fs(fnames, temp_f):
    #returns a list of Optf objects given a list of fnames.
    optfs = []
    for f in fnames:
        dmobj = Optf(f)
        dmobj.load()
        dmobj.load_filts(temp_f)
        dmobj.psd = np.ones(1023) 
        dmobj.fit()
        dmobj.saveOptf(clear_data = True)
        optfs.append(dmobj)
  
    return optfs

def load_fs(fnames):
    #loads a list of optf objects from a file
    optfs = []
    for f in fnames:
        dmobj = Optf(f)
        dmobj.loadOptf()
        optfs.append(dmobj)
  
    return optfs    

