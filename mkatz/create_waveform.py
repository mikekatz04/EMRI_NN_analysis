import numpy as np 
import sys
import pdb
import time
import subprocess
import h5py as h5
import os
from shutil import copyfile
import multiprocessing as mp
import h5py


class EMRIWaveformCreation:
	"""
	general equations for analysis of EMRI waveforms
	"""
	def run_waveform(ecc=0.1, inc=0.1, p=8.0, bh_mass=1e6, a=0.8, dist=1.0, co_mass=1e1, wave_type='AAK', filepath='temp', extension='', return_data=False, remove=False, extension_2=''):

		"""
		creates a waveform for an EMRI by running the kludge generator by Chua et al 2017.
		"""
		with open('SetPar_Template' + extension, 'r') as f_in:
			lines = f_in.readlines()

		#line 6 is file name
		#line 57 is the inclination
		#line 54 is initial eccentricity
		lines[6]  = filepath + extension + extension_2 + '\n'
		#lines[6]  = 'NN_test_fund_freqs'

		#adjust bh_mass
		lines[54] = '%.18e\n'%bh_mass

		#adjust co_mass
		lines[51] = '%.18e\n'%co_mass

		#adjust p 
		lines[33] = '%.18e\n'%p

		#adjust ecc
		lines[60] = '%.18e\n'%ecc

		#adjust a
		lines[57] = '%.18e\n'%a

		#adjust inc 
		lines[63] = '%.18e\n'%inc

		#adjust dist
		lines[90] = '%.18e\n'%dist

		out_string = ''.join(lines)
		with open('SetPar_Template' + extension, 'w') as f_out:
			f_out.write(out_string)
		#print('start_sub')
		subprocess.run("bin/" + wave_type + "_waveform SetPar_Template" + extension, shell=True)
		#print('finish_sub')


		if return_data:
			data = np.genfromtxt(filepath + extension + extension_2 + '_' + 'wave' + '.dat').T

		if remove:
			os.remove(filepath + extension + extension_2 + '_' + 'wave' + '.dat')

		if return_data:
			return data

		return

	def find_spectrogram(t, sig, segment_time=12.0):
		"""
		Takes a time domain signal of an EMRI and finds custom spectrogram by breaking it up into segment_time (hours) length segments. Computes DFT with np.fft
		"""
		dt = t[1]-t[0]
		length_of_signal = t[-1]

		segment_time = segment_time*3600.00
		segment_times = np.arange(segment_time, length_of_signal, segment_time)
		indices_of_segments = np.searchsorted(t, segment_times, side='left')
		segments = np.split(sig, indices_of_segments)
		segments = np.vstack(segments[0:-1])

		#freqs = np.fft.rfftfreq(segments.shape[1], d=dt)
		freqs = np.fft.rfftfreq(segments.shape[1], d=dt)

		freqs = freqs*(freqs>=0.0) + -1*freqs*(freqs<0.0)

		keep = np.where((freqs>=5e-5) & (freqs <=1e-2))[0]

		#spectrogram = np.abs(np.fft.rfft(segments, axis=-1))[:,keep]
		spectrogram = np.abs(np.fft.fft(segments, axis=-1))[:,keep]
		times = np.arange(len(segment_times))*segment_time+segment_time/2.
		return spectrogram, times, freqs[keep]

def parallel_func(j, length, kwargs):

	spec_out = []
	hp_out = []
	ecc_out = []
	inc_out = []
	for i in range(length): 
		temp_kwargs = {key:kwargs[key][i] for key in kwargs}
		ecc_out.append(kwargs['ecc'][i])
		inc_out.append(kwargs['inc'][i])

		temp_kwargs['extension'] = '_' + str(j)
		#temp_kwargs['remove']  = True
		temp_kwargs['return_data'] = True
		temp_kwargs['filepath'] = 'chirp_test'
		temp_kwargs['wave_type'] = 'NK'
		temp_kwargs['extension_2'] = str(i)

		wave = EMRIWaveformCreation.run_waveform(**temp_kwargs)

		spec, t, f = EMRIWaveformCreation.find_spectrogram(wave[0], wave[1] - 1j*wave[2])

		spec_out.append(spec)
		hp_out.append(wave[1])
		if i == 0:
			t_spec_out = t
			f_out = f
			t_out = wave[0]

	spec_out = np.asarray(spec_out)
	hp_out = np.asarray(hp_out)
	ecc_out = np.asarray(ecc_out)
	inc_out = np.asarray(inc_out)
	return {'spec':spec_out, 'hp':hp_out, 't_spec':t_spec_out, 'f':f_out,'t':t_out, 'ecc': ecc_out, 'inc': inc_out}


class ParallelGeneration:
	"""
	Generates waveforms in parallel. 
	"""

	def __init__(self, num_splits=100, num_processors=None):
		self.num_splits = num_splits
		if num_processors == None:
			self.num_processors = mp.cpu_count()

		else:
			self.num_processors = num_processors

		#make sure there is an input file for each processor
		for i in range(self.num_processors):
			copyfile('SetPar_Template', 'SetPar_Template' + '_' + str(i))

	def prepare_parallel(self, length, **kwargs):
		#set up inputs for each processor
		#based on num_splits which indicates max number of boxes per processor
		self.kwargs = kwargs
		
		split_val = int(np.ceil(length/self.num_splits))
		split_inds = [self.num_splits*i for i in np.arange(1,split_val)]

		inds_split = np.split(np.arange(length), split_inds)
		self.args = []
		for i, ind_split in enumerate(inds_split):
			self.args.append((i,len(ind_split)) + ({key:kwargs[key][ind_split] for key in kwargs},))
		


		return

	def run_parallel(self):
		print('Total Processes: ',len(self.args))

		#test parallel func
		#pdb.set_trace()
		#check = [parallel_func(*self.args[i]) for i in [0,1]]
		#pdb.set_trace()

		para_start_time = time.time()
		results = []
		print('numprocs', self.num_processors)
		with mp.Pool(self.num_processors) as pool:
			print('start pool\n')
			results = [pool.apply_async(parallel_func, arg) for arg in self.args]
			out = [r.get() for r in results]

		self.trans = {key: np.concatenate([out1[key] for out1 in out], axis=0) for key in ['spec','hp','ecc','inc']}

		for key in ['t', 't_spec','f']:
			self.trans[key] = out[0][key]
		return

	def write_to_file(self, filename):
		with h5py.File(filename, 'w') as f_out:

			for key in self.trans:
				f_out.create_dataset(key, data = self.trans[key] , dtype = self.trans[key].dtype.name, chunks = True, compression = 'gzip', compression_opts = 9)
		return

	def remove_files(self):
		for i in range(self.num_processors):
			os.remove('SetPar_Template' + '_' + str(i))
		return


def generate_grid_files():

	
	#for grid generation
	st = time.time()

	M_in = np.logspace(4, 7, 4)
	a_in = np.linspace(0.5, 0.9, 4)
	M_in, a_in = np.meshgrid(M_in, a_in)
	M_in, a_in = M_in.ravel(), a_in.ravel()

	p_in = np.linspace(8.0, 15.0, 2)
	ecc_in = np.linspace(0.1, 0.7, 2)
	inc_in = np.linspace(0.2, 60*np.pi/180, 2)
	#ecc_in, inc_in = np.meshgrid(ecc_in, inc_in)#, p_in)
	#ecc_in, inc_in = ecc_in.ravel(), inc_in.ravel()


	para = ParallelGeneration(num_splits=2)
	

	#i=0

	#for random generation
	#number = 100
	#ecc_in = np.random.uniform(low=0.01, high=0.4, size=number)
	#inc_in = np.random.uniform(low=0.02, high=45*np.pi/180, size=number)

	for i in range(2):
		for num, M in enumerate(M_in):

			para.prepare_parallel(len(ecc_in), **{'ecc':np.full((len(a_in),),ecc_in[i]), 'bh_mass':np.full((len(a_in),),M), 'inc':np.full((len(a_in),),inc_in[i]), 'p':np.full((len(a_in),),p_in[i]), 'a':a_in})
			para.run_parallel()
			para.write_to_file('spectrograms_full_{}.hdf5'.format(i*1000 + num))

			print('\n\n', M, '\n\n')

	print('time:', time.time()-st)

	para.remove_files()

	return


if __name__ == "__main__":
	generate_grid_files()


