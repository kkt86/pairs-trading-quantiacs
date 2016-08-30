import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.stattools import adfuller


#================================== DEFINE PAIRS CLASS ==================================

class Pairs:
	
	def __init__(self,data):
		
		### Extract date from data
		self.date = data.index[-1]

		### Set initial position to NONE
		self.position = 'NONE'

		### set open, close and limit values (apply on the z-score)
		self.open_high = 2.0
		self.close_high = 0.0
		self.limit_high = 3.5

		self.open_low = -2.0
		self.close_low = 0.0
		self.limit_low = -3.5

		self.lookback = 50

		self.position = 'NONE'
		self.p = np.array([0., 0.])
		self.prices = np.array([0., 0.])
		self.max_drawdown = -0.1
		self.value_portfolio = 0.0

		### Compute beta and hurst exponent
		self.beta = self.compute_beta(data)
		self.hurst = self.compute_hurst(data)

	def compute_beta(self,data):
		"""
		Computes the regression coefficient beta, provided the data.
		"""
		from scipy import stats
		beta = stats.linregress(data.ix[-self.lookback:,0],data.ix[-self.lookback:,1])[0]
		return beta

	def update_beta(self, data):
		"""
		Updates beta inside the class
		"""
		self.beta = self.compute_beta(data)

	def update_date(self,data):
		"""
		Updats the current date
		"""
		self.date = data.index[-1]

	def compute_hurst(self,data):
		"""
		Returns the Hurst Exponent of the time series vector ts
		"""

		### get mean reverting time series
		ts = data.ix[:,1] - self.beta*data.ix[:,0]

		# Create the range of lag values
		lags = range(2,100)
    
		# Calculate the array of the variances of the lagged differences
		tau = [np.sqrt(np.std(np.subtract(ts[lag:],ts[:-lag]))) for lag in lags]

		# Use a linear fit to estimate the Hurst Exponent
		poly = np.polyfit(np.log(lags), np.log(tau),1)

		return poly[0]*2  

	def update_hurst(self, data):
		"""
		Updates the Hurst exponent inside the class.
		"""
		self.hurst = self.compute_hurst(data)

	def compute_lambda(self,data):
		"""
		Computes the lambda coefficient in: dy(t) = lambda * y(t-1)
		"""
		ts = data.ix[:,1] - self.beta*data.ix[:,0]
		poly = np.polyfit(ts[:-1],np.diff(ts),1)
		return poly[0]

	def compute_halflife(self,data):
		lmbda = self.compute_lambda(data)
		if lmbda < 0:
			half_life = -np.log(2)/lmbda
		else:
			half_life = 0
		return half_life

	def update_lookback(self, data):
		"""
		Updates the lookback according to the halflife of the pair.
		"""
		halflife = self.compute_halflife(data)
		self.lookback = np.int64(np.maximum(1.5*halflife,30))

	def update_all(self, data):
		"""
		Updates the hedge coefficient beta, the Hurst exponennt and the lookback.
		"""
		self.update_beta(data)
		self.update_hurst(data)
		self.update_lookback(data)

	def compute_zscore(self,data):
		"""
		Returns the zscore of the two series as new time series.
		"""

		#self.update_lookback(data)
		#self.update_beta(data)

		x = data[data.columns[0]]
		y = data[data.columns[1]]
		z = y - self.beta*x

		z_mean = z.rolling(self.lookback).mean()
		z_std = z.rolling(self.lookback).std()

		return (z - z_mean)/z_std

	def compute_adf_pvalue(self, data):
		"""
		Returns the p-value of the ADF test
		"""
		try:
			zscore = self.compute_zscore(data)
			p_value = adfuller(zscore[-self.lookback:],1)[1]
			return p_value
		except:
			return 1


	def compute_drawdown(self, data):
		"""
		Computes the drawdown of portfolio
		"""
		drawdown = np.sum(np.sign(self.p)*(data.ix[-1,:] - self.prices)/self.prices)
		return drawdown

	def open_signal(self,data):
		"""
		Returns type of entry signal.
		"""

		### update lookback period and beta and compute the zscore
		#self.update_lookback(data)
		#self.update_beta(data)
		zscore = self.compute_zscore(data)[-1]

		### get p-value of the ADF test
		#adf_test = (self.compute_adf_pvalue(data) < 0.05)
		

		### evaluate zscore and return entry signal
		if zscore >= self.open_high:
			return 'ENTER_ABOVE'
		elif zscore <= self.open_low:
			return 'ENTER_BELOW'
		else:
			return 'NONE'

	def close_signal(self, data):
		"""
		Returns close signal, if position is to be closed.
		"""

		### update lookback period and beta and compute the zscore
		#self.update_lookback(data)
		#self.update_beta(data)
		zscore = self.compute_zscore(data)[-1]

		if self.position == 'ENTERED_ABOVE' and zscore <= self.close_high:
			return 'CLOSE'
		
		elif self.position == 'ENTERED_BELOW' and zscore >= self.close_low:
			return 'CLOSE'

		elif self.compute_drawdown(data) <= self.max_drawdown:
			return 'CLOSE_MAX_DD_EXCEEDED'
		
		else:
			return 'NONE'



	def get_positions(self, data):
		"""
		Computes positions for the two assets
		"""

		### get old values of p
		p = self.p

		### If no capital is invested, get open signal
		if self.position == 'NONE':
			open_signal = self.open_signal(data)

			if open_signal == 'ENTER_ABOVE':
				p = np.array([self.beta, -1.])
				self.position = 'ENTERED_ABOVE'
				self.p = p
				self.prices = data.ix[-1,:]

			if open_signal == 'ENTER_BELOW':
				p = np.array([-self.beta, 1.])
				self.position = 'ENTERED_BELOW'
				self.p = p
				self.prices = data.ix[-1,:]

		### If position already entered, evaluate exit signals
		if self.position != 'NONE':
			close_signal = self.close_signal(data)

			if self.position == 'ENTERED_ABOVE' and close_signal == 'CLOSE':
				p = np.array([0., 0.])
				self.position = 'NONE'
				self.p = p
				self.prices = np.array([0., 0.])

			if self.position == 'ENTERED_BELOW' and close_signal == 'CLOSE':
				p = np.array([0., 0.])
				self.position = 'NONE'
				self.p = p
				self.prices = np.array([0., 0.])

			if close_signal == 'CLOSE_MAX_DD_EXCEEDED':
				p = np.array([0., 0.])
				self.position = 'MAX_DD_EXCEEDED'
				self.p = p
				self.prices = np.array([0., 0.])

			if self.position == 'MAX_DD_EXCEEDED':
				zscore = self.compute_zscore(data)[-1]
				if np.abs(zscore) <= 0.5:
					self.position = 'NONE'
					p = self.p


		return p










# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)

def mySettings():

	settings={}


	settings['markets'] = ['CASH', 'JPM', 'SCHW', \
							'ETFC', 'HBAN', \
							'AAPL', 'APH', \
							'FB', 'QCOM', \
							'HPQ', 'FOXA', \
							'GGP', 'MAC', \
							'MRK', 'MJN', \
							'AET', 'HUM', \
							'AME', 'HRS', \
							'HAL', 'NBR', \
							'CMS', 'XEL', \
							'D', 'PSX', \
							'DUK', 'FE']


	settings['slippage']    = 0.05
	settings['budget']      = 1000000
	settings['beginInSample'] = '20130101'
	settings['endInSample']   = '20160101'
	settings['lookback']    = 256

	settings['pairs'] = {}


	return settings



def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, settings):


	### Preprocess stock data
	data_JPM_SCHW = preprocess_data(DATE, CLOSE[:,1:3])
	data_ETFC_HBAN = preprocess_data(DATE, CLOSE[:,3:5])
	data_AAPL_APH = preprocess_data(DATE, CLOSE[:,5:7])
	data_FB_QCOM = preprocess_data(DATE, CLOSE[:,7:9])
	data_HPQ_FOXA = preprocess_data(DATE, CLOSE[:,9:11])
	data_GGP_MAC = preprocess_data(DATE, CLOSE[:,11:13])
	data_MRK_MJN = preprocess_data(DATE, CLOSE[:,13:15])
	data_AET_HUM = preprocess_data(DATE, CLOSE[:,15:17])
	data_AME_HRS = preprocess_data(DATE, CLOSE[:,17:19])
	data_HAL_NBR = preprocess_data(DATE, CLOSE[:,19:21])
	data_CMS_XEL = preprocess_data(DATE, CLOSE[:,21:23])
	data_D_PSX = preprocess_data(DATE, CLOSE[:,23:25])
	data_DUK_FE = preprocess_data(DATE, CLOSE[:,25:27])


	### Assign pairs into settings
	if len(settings['pairs']) == 0:
		settings['pairs']['JPM_SCHW'] = Pairs(data_JPM_SCHW)
		settings['pairs']['ETFC_HBAN'] = Pairs(data_ETFC_HBAN)
		settings['pairs']['AAPL_APH'] = Pairs(data_AAPL_APH)
		settings['pairs']['FB_QCOM'] = Pairs(data_FB_QCOM)
		settings['pairs']['HPQ_FOXA'] = Pairs(data_HPQ_FOXA)
		settings['pairs']['GGP_MAC'] = Pairs(data_GGP_MAC)
		settings['pairs']['MRK_MJN'] = Pairs(data_MRK_MJN)
		settings['pairs']['AET_HUM'] = Pairs(data_AET_HUM)
		settings['pairs']['AME_HRS'] = Pairs(data_AME_HRS)
		settings['pairs']['HAL_NBR'] = Pairs(data_HAL_NBR)
		settings['pairs']['CMS_XEL'] = Pairs(data_CMS_XEL)
		settings['pairs']['D_PSX'] = Pairs(data_D_PSX)
		settings['pairs']['DUK_FE'] = Pairs(data_DUK_FE)


	### Update parameters for each pair (on a daily basis)
	if toDate(DATE[-1]).day == 1:
		settings['pairs']['JPM_SCHW'].update_all(data_JPM_SCHW)
		settings['pairs']['ETFC_HBAN'].update_all(data_ETFC_HBAN)
		settings['pairs']['AAPL_APH'].update_all(data_AAPL_APH)
		settings['pairs']['FB_QCOM'].update_all(data_FB_QCOM)
		settings['pairs']['HPQ_FOXA'].update_all(data_HPQ_FOXA)
		settings['pairs']['GGP_MAC'].update_all(data_GGP_MAC)
		settings['pairs']['MRK_MJN'].update_all(data_MRK_MJN)
		settings['pairs']['AET_HUM'].update_all(data_AET_HUM)
		settings['pairs']['AME_HRS'].update_all(data_AME_HRS)
		settings['pairs']['HAL_NBR'].update_all(data_HAL_NBR)
		settings['pairs']['CMS_XEL'].update_all(data_CMS_XEL)
		settings['pairs']['D_PSX'].update_all(data_D_PSX)
		settings['pairs']['DUK_FE'].update_all(data_DUK_FE)


	### Assign portion of capital on each asset
	p = np.zeros(len(settings['markets'])) # ... initialize p
	

	p[0] = 1.

	p[1:3] = settings['pairs']['JPM_SCHW'].get_positions(data_JPM_SCHW)
	p[3:5] = settings['pairs']['ETFC_HBAN'].get_positions(data_ETFC_HBAN)
	p[5:7] = settings['pairs']['AAPL_APH'].get_positions(data_AAPL_APH)
	p[7:9] = settings['pairs']['FB_QCOM'].get_positions(data_FB_QCOM)
	p[9:11] = settings['pairs']['HPQ_FOXA'].get_positions(data_HPQ_FOXA)
	p[11:13] = settings['pairs']['GGP_MAC'].get_positions(data_GGP_MAC)
	p[13:15] = settings['pairs']['MRK_MJN'].get_positions(data_MRK_MJN)
	p[15:17] = settings['pairs']['AET_HUM'].get_positions(data_AET_HUM)
	p[17:19] = settings['pairs']['AME_HRS'].get_positions(data_AME_HRS)
	p[19:21] = settings['pairs']['HAL_NBR'].get_positions(data_HAL_NBR)
	p[21:23] = settings['pairs']['CMS_XEL'].get_positions(data_CMS_XEL)
	p[23:25] = settings['pairs']['D_PSX'].get_positions(data_D_PSX)
	p[25:27] = settings['pairs']['DUK_FE'].get_positions(data_DUK_FE)


	if np.any(p[1:] != 0.): p[0] = 0.0 # ... set CASH to 0 if capital invested somewhere else
	
	settings['p'] = p # ... save p


	return p, settings





################################# MY FUNCTIONS #################################

def toDate(x):
	"""
	Converts integer provided as DATE from Quantiacs data to datetime
	"""
	return datetime.datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:8]))

def preprocess_data(DATE,CLOSE):
	"""
	Returns a prerpocessed data for the selected pair	
	"""
	data = pd.DataFrame({'DATE' : DATE,'x_t' : CLOSE[:,0], 'y_t' : CLOSE[:,1]}) # ... create dataframe with the pair of assets
	data['DATE'] = data['DATE'].apply(toDate)
	data = data.set_index('DATE')
	data = data.dropna(axis=0,how='any') # .. drop NaN
	return data















