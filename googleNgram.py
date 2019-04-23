### Google Ngram Data Tools
## Alex John Quijano
## Created: 9/13/2017
## Updated: 1/31/2018

import os
import re
import sys
import math
import errno
import numpy as np
import subprocess as sb

# savitzky golay smoothing function
def savitzky_golay(y, window_size, order, deriv=0, rate=1):

	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError or msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

# Read Google ngram data
def read(n,sT,l,download=False,smoothed=False,restriction=True,annotations=False,ignore_case=True,specific_fileName='all'):

	if download == True:
		prompt_user = input('You are about to download and normalize the Google '+n+'gram '+l+
												' dataset! This might take a while. Please confirm [y/n]: ')
		if prompt_user != 'y':
			print('Terminated!')
			try:
				sys.exit()
			except SystemExit:
				sys.exit
		elif prompt_user == 'y':
			try:
				filteredData_fileName = n+'gram-filtered/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered'
				f_temp = open(filteredData_fileName)
				print(filteredData_fileName+' already exists!')
			except FileNotFoundError:
				default_yearBounds = ['1900','2008']
				default_excludeAnnotations = '1'
				default_bookCountLowerBound = {'eng':'500','eng-us':'500','eng-gb':'500','eng-fiction':'200' ,
																			 'chi-sim':'1','fre':'200','ger':'200','heb':'10','ita':'200','rus':'200','spa':'200'}
				print('Processing with default parameters...')
				print('\tn: '+n)
				print('\tlanguage: '+l)
				if n != '1':
					if specific_fileName == '':
						specific_fileName = input('Please enter file name of specific 1gram list for downloading '+n+'grams unless you want all data then just press enter: ')
				print('\tspecific 1grams: '+specific_fileName)
				print('\tyear lower bound: '+default_yearBounds[0])
				print('\tyear upper bound: '+default_yearBounds[1])
				print('\texclude annotations: 1')
				print('\tbook count lower bound: '+default_bookCountLowerBound[l])
				sb.call('./downloadAndFilter.ngram.sh '+n+' '+l+
								' '+default_yearBounds[0]+' '+default_yearBounds[1]+
								' 1 '+default_bookCountLowerBound[l],
								shell=True)
			try:
				normalizedData_fileName = n+'gram-normalized/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered.R'+str(restriction)+'A'+str(annotations)+'I'+str(ignore_case)+'.'+specific_fileName+'.'+sT+'.npy'
				f_temp = open(normalizedData_fileName)
				print(normalizedData_fileName+' already exist!')
			except FileNotFoundError:
				print('Normalizing...')
				sb.call('./normalize.ngram.py '+n+' '+l+' '+
								str(restriction)+' '+str(annotations)+' '+str(ignore_case)+' '+specific_fileName,
								shell=True)
			print('Yay! Completed!')
	elif download == False:
		pass

	try:
		# raw data (in of directory)
		matrix_path = '/google-ngram/'+n+'gram-normalized/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered.R'+str(restriction)+'A'+str(annotations)+'I'+str(ignore_case)+'.'+specific_fileName+'.'+sT+'.npy'
		matrix_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+matrix_path
		pos_path = '/google-ngram/'+n+'gram-normalized/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered.R'+str(restriction)+'A'+str(annotations)+'I'+str(ignore_case)+'.'+specific_fileName+'.pos.npy'
		pos_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+pos_path
		vocabulary_path = '/google-ngram/'+n+'gram-normalized/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered.R'+str(restriction)+'A'+str(annotations)+'I'+str(ignore_case)+'.'+specific_fileName+'.vocabulary.npy'
		vocabulary_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+vocabulary_path

		dataMatrix_default = np.load(matrix_directory)
		vocabulary = np.load(vocabulary_directory).item()
		pos = np.load(pos_directory)

	except FileNotFoundError:
		try:
			# raw data (out of directory)
			matrix_path = '/raw-data/google-ngram/'+n+'gram-normalized/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered.R'+str(restriction)+'A'+str(annotations)+'I'+str(ignore_case)+'.'+specific_fileName+'.'+sT+'.npy'
			matrix_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+matrix_path
			pos_path = '/raw-data/google-ngram/'+n+'gram-normalized/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered.R'+str(restriction)+'A'+str(annotations)+'I'+str(ignore_case)+'.'+specific_fileName+'.pos.npy'
			pos_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+pos_path
			vocabulary_path = '/raw-data/google-ngram/'+n+'gram-normalized/googlebooks-'+l+'-all-'+n+'gram-20120701.filtered.R'+str(restriction)+'A'+str(annotations)+'I'+str(ignore_case)+'.'+specific_fileName+'.vocabulary.npy'
			vocabulary_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+vocabulary_path

			dataMatrix_default = np.load(matrix_directory)
			vocabulary = np.load(vocabulary_directory).item()
			pos = np.load(pos_directory)

		except FileNotFoundError:
			print('Error: The computed-data directory can not be found or '+sT+' '+n+'gram dataset for '+l+' does not exist anywhere.')
			print('Please refer to the README.md file.')
			try:
				sys.exit()
			except SystemExit:
				sys.exit

	if smoothed == True:
		data_Matrix_default_smoothed = np.zeros(dataMatrix_default.shape,dtype=float)
		for i in range(dataMatrix_default.shape[0]):
			data_Matrix_default_smoothed[i,:] = savitzky_golay(dataMatrix_default[i,:], 11, 3) # window size 11, polynomial order 3
		return np.array(dataMatrix_default,dtype=float), vocabulary, pos, np.array(data_Matrix_default_smoothed)
	else:
		return np.array(dataMatrix_default,dtype=float), vocabulary, pos

# Get total raw counts of each year
def year_counts(n,l):

	# Read raw score data of a given language
	rscore, vocabulary, pos = read(n=n,sT='rscore',l=l)

	# Count total rscores for each year
	year_counts = []
	for i in range(0,rscore.shape[1]):
		year_counts.append(np.sum(rscore[:,i]))
	year_counts = np.array(year_counts,dtype=int)

	# Compute vocabulary size count ratio for each year
	year_ratio = np.divide(len(vocabulary),year_counts)

	return year_counts, year_ratio

# Get total raw counts of each ngram
def ngram_counts(n,l):

	# Read raw score data of a given language
	rscore, vocabulary, pos = read(n=n,sT='rscore',l=l)

	# Count total rscores for each ngram
	ngram_counts = []
	for i in range(0,rscore.shape[0]):
		ngram_counts.append(np.sum(rscore[i,:]))
	ngram_counts = np.array(ngram_counts,dtype=int)

	return ngram_counts

# Get probablities and ranks
def zipf(n,l,return_index=False):

	# Ngram counts
	n_counts = ngram_counts(n,l)

	# Sort and compute pmf
	ngram_counts_sorted_index = np.argsort(-1*n_counts)
	ngram_counts_sorted = n_counts[ngram_counts_sorted_index]

	# Probabilities and ranks
	ngram_pmf = np.divide(ngram_counts_sorted,sum(ngram_counts_sorted))
	ngram_ranks = range(1,len(ngram_pmf)+1)
	ngram_cdf = []
	i_cumulative = 0
	for i in ngram_pmf:
		i_cumulative = i_cumulative + i
		ngram_cdf.append(i_cumulative)

	return ngram_pmf, ngram_cdf, ngram_ranks, ngram_counts_sorted_index

# stop words - get indices of ngrams with stop words
def stopWord_getIndex(n,l):

	# defined stop words
	stop_words = []
	try:
		file = open(n+'gram-list/'+n+'gram-list-stop-word-'+l)
	except FileNotFoundError:
		file = open(n+'gram-list/'+l+'/'+n+'gram-list-stop-word-'+l)
	for f in file:
		stop_words.append(f.replace('\n',''))

	# read raw score data of a given language
	rscore, vocabulary, pos = read(n=n,sT='rscore',l=l)

	ngrams = vocabulary['forward']
	index_out = []
	for n in ngrams:
		n_split = n.split(' ')
		for ns in n_split:
			if ns in stop_words:
				index_out.append(ngrams[n])

	return index_out

# get indices of ngrams with a given word
def ngram_getIndex(n,l,word,exclude_stopWord=False):

	# read raw score data of a given language
	rscore, vocabulary, pos = read(n=n,sT='rscore',l=l)

	ngrams = vocabulary['forward']
	index_out = []

	if n == '1':
		for w in word:
			try:
				index_out.append(ngrams[w])
			except:
				pass
	else:
		for b in ngrams.keys():
			b_split = b.split(' ')
			for w in word:
				if w in b_split:
					try:
						index_out.append(ngrams[b])
					except KeyError:
						pass

	if exclude_stopWord == True:
		stopWords_index = stopWords_getIndex(n=n,l=l)
		index_sw = []
		for sw in stopWords_index:
			if sw in index_out:
				index_sw.append(sw)
				index_out.remove(sw)
		return index_out, index_sw
	else:
		return index_out