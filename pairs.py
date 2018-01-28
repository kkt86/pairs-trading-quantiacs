import numpy as np
import pandas as pd
import datetime


np.arange


#================================== DEFINE PAIRS CLASS ====================

class Pairs:

	def __init__(self, data):
		# Extract date from data
		self.date = data.index[-1]

		### Set initial position to NONE
		self.position = 'NONE'

		### set open, close and limit values (apply on the z-score)
		self.open_high = 2.0
		self.close_high = 0.0

		self.open_low = -2.0
		self.close_low = 0.0

		self.lookback = 50

		self.position = 'NONE'
		self.p = np.array([0., 0.])
		self.max_drawdown = -0.2
		self.max_value_portfolio = np.nan

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

		x = data[data.columns[0]]
		y = data[data.columns[1]]
		z = y - self.beta*x

		z_mean = z.rolling(self.lookback).mean()
		z_std = z.rolling(self.lookback).std()

		return (z - z_mean)/z_std


	def compute_drawdown(self, data):
		"""
		Computes the drawdown of portfolio
		"""

		### update max_value_portfolio
		weights = self.p/np.sum(np.abs(self.p))
		value_portfolio = np.sum(weights*data.ix[-1,:])
		self.max_value_portfolio = np.maximum(self.max_value_portfolio, value_portfolio)

		drawdown = (value_portfolio - self.max_value_portfolio)/self.max_value_portfolio
		return drawdown

	def open_signal(self,data):
		"""
		Returns type of entry signal.
		"""

		### update lookback period and beta and compute the zscore
		zscore = self.compute_zscore(data)[-1]
		

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
		zscore = self.compute_zscore(data)[-1]

		self.compute_drawdown(data)

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
				weights = self.p/np.sum(np.abs(self.p))
				self.max_value_portfolio = np.sum(weights*data.ix[-1,:])

			if open_signal == 'ENTER_BELOW':
				p = np.array([-self.beta, 1.])
				self.position = 'ENTERED_BELOW'
				self.p = p
				weights = self.p/np.sum(np.abs(self.p))
				self.max_value_portfolio = np.sum(weights*data.ix[-1,:])

		### If position already entered, evaluate exit signals
		if self.position != 'NONE':
			close_signal = self.close_signal(data)

			if self.position == 'ENTERED_ABOVE' and close_signal == 'CLOSE':
				p = np.array([0., 0.])
				self.position = 'NONE'
				self.p = p
				#self.prices = np.array([0., 0.])
				self.max_value_portfolio = np.nan

			if self.position == 'ENTERED_BELOW' and close_signal == 'CLOSE':
				p = np.array([0., 0.])
				self.position = 'NONE'
				self.p = p
				#self.prices = np.array([0., 0.])
				self.max_value_portfolio = np.nan

			if close_signal == 'CLOSE_MAX_DD_EXCEEDED':
				p = np.array([0., 0.])
				self.position = 'MAX_DD_EXCEEDED'
				self.p = p
				#self.prices = np.array([0., 0.])
				self.max_value_portfolio = np.nan

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

	### Cointegrating banks
	settings['markets']      = ['CASH', 'JPM', 'SCHW'] 	# ==> SR:  0.7553  
	#settings['markets']      = ['CASH', 'MS', 'STT'] 		# ==> SR:  0.7094
	#settings['markets']      = ['CASH', 'ETFC', 'HBAN'] 	# ==> SR:  1.4802
	
	### Cointegrating tech companies
	#settings['markets']      = ['CASH', 'AAPL', 'APH'] 	# ==> SR:  0.6581
	#settings['markets']      = ['CASH', 'FB', 'QCOM'] 		# ==> SR:  1.2483
	#settings['markets']      = ['CASH', 'FB', 'FLIR'] 		# ==> SR:  0.9742
	#settings['markets']      = ['CASH', 'HPQ', 'FOXA'] 	# ==> SR:  1.1262
	#settings['markets']      = ['CASH', 'HPQ', 'CTL'] 		# ==> SR: -0.1811
	#settings['markets']      = ['CASH', 'QCOM', 'VRSN'] 	# ==> SR: -0.3518
	#settings['markets']      = ['CASH', 'CRM', 'PBI'] 		# ==> SR:  0.5599
	#settings['markets']      = ['CASH', 'FLIR', 'NWSA'] 	# ==> SR:  0.7793
	#settings['markets']      = ['CASH', 'FLIR', 'VIAB'] 	# ==> SR: -1.2582
	#settings['markets']      = ['CASH', 'JNPR', 'XRX'] 	# ==> SR:  0.2037

	### Cointegrating aerospace companies
	#settings['markets']      = ['CASH', 'HON', 'NOC'] 		# ==> SR: -0.5076

	### Cointegrating Oil-Gas companies
	#settings['markets']      = ['CASH', 'DNR', 'NBL'] 		# ==> SR:  0.1246

	### Cointegrating Real Estate companies
	#settings['markets']      = ['CASH', 'GGP', 'MAC'] 		# ==> SR:  1.9376

	### Cointegrating healthcare companies
	#settings['markets']      = ['CASH', 'MDT', 'ZTS'] 		# ==> SR: -0.9783
	#settings['markets']      = ['CASH', 'MRK', 'MJN'] 		# ==> SR:  0.6890
	#settings['markets']      = ['CASH', 'AET', 'HUM'] 		# ==> SR:  0.6733
	#settings['markets']      = ['CASH', 'AGN', 'ZTS'] 		# ==> SR: -0.4765
	#settings['markets']      = ['CASH', 'CAH', 'ZTS'] 		# ==> SR: -0.6124
	#settings['markets']      = ['CASH', 'VRTX', 'ZTS'] 	# ==> SR: -0.6292


	### Cointegrating industries
	#settings['markets']      = ['CASH', 'EMR', 'MAS'] 		# ==> SR: -0.0769
	#settings['markets']      = ['CASH', 'HON', 'BMS'] 		# ==> SR: -0.4541
	#settings['markets']      = ['CASH', 'LMT', 'FLR'] 		# ==> SR:  0.2849
	#settings['markets']      = ['CASH', 'AME', 'GRMN'] 	# ==> SR:  -0.3688
	#settings['markets']      = ['CASH', 'AME', 'GT'] 		# ==> SR: -0.2889
	#settings['markets']      = ['CASH', 'AME', 'HRS'] 		# ==> SR:  0.5482
	#settings['markets']      = ['CASH', 'AME', 'KSU'] 		# ==> SR:  0.0899
	#settings['markets']      = ['CASH', 'AME', 'MLM'] 		# ==> SR: -0.1210
	#settings['markets']      = ['CASH', 'AME', 'PKI'] 		# ==> SR:  0.3429
	#settings['markets']      = ['CASH', 'AME', 'TYC'] 		# ==> SR:  0.0938
	#settings['markets']      = ['CASH', 'AME', 'VMC'] 		# ==> SR: -0.0856

	### Cointegrating energy companies
	#settings['markets']      = ['CASH', 'COP', 'EOG'] 		# ==> SR:  0.2611
	#settings['markets']      = ['CASH', 'XOM', 'CHK'] 		# ==> SR: -0.7106
	#settings['markets']      = ['CASH', 'HAL', 'NBR'] 		# ==> SR: 0.2698
	#settings['markets']      = ['CASH', 'CMS', 'XEL'] 		# ==> SR: 1.6298
	#settings['markets']      = ['CASH', 'D', 'DUK'] 		# ==> SR:  0.0612
	#settings['markets']      = ['CASH', 'D', 'FE'] 		# ==> SR: 0.1039
	#settings['markets']      = ['CASH', 'D', 'PSX'] 		# ==> SR:  0.9998
	#settings['markets']      = ['CASH', 'DUK', 'FE'] 		# ==> SR:  1.1402
	#settings['markets']      = ['CASH', 'HP', 'NE'] 		# ==> SR: -0.0634
	#settings['markets']      = ['CASH', 'NEE', 'EOG'] 		# ==> SR: -0.2656





	settings['slippage']    = 0.05
	settings['budget']      = 1000000
	settings['beginInSample'] = '20130101'
	settings['endInSample']   = '20160101'
	settings['lookback']    = 256

	settings['pairs'] = {}


	return settings



def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, settings):


	### Load stock data and assure that dates match
	data = preprocess_data(DATE, CLOSE[:,1:3])




	### Assign pair into settings
	if not settings['pairs'].has_key('pair1'):
		pair1 = Pairs(data)
		settings['pairs']['pair1'] = pair1



	### Update parameters once a month
	if toDate(DATE[-1]).day == 1: 
		settings['pairs']['pair1'].update_all(data)



	### Assign investments
	p = np.array([1., 0., 0.]) # ... initialize p
	p[1:3] =  settings['pairs']['pair1'].get_positions(data) # 
	if np.any(p[1:] != 0.): p[0] = 0.0 # ... set CASH to 0 if capital invested
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







