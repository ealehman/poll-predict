#!/usr/bin/env python
#poll_data.py
 

from fnmatch import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pattern import web

from matplotlib import rcParams
import re
import random

#colors from colorbrewer2.org 
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]
#rcParams settings code from CS109
rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'

#retrieves xml for RCP data given the poll id
def get_poll_xml(poll_id):
    xml = 'http://charts.realclearpolitics.com/charts/' + str(poll_id) + '.xml'
    return requests.get(xml).text

#CS109
def _strip(s):
	return re.sub(r'[\W_]+', '', s)

#code provided by CS109 
def plot_colors(xml):
    dom = web.Element(xml)
    result = {}
    for graph in dom.by_tag('graph'):
        title = _strip(graph.attributes['title'])
        result[title] = graph.attributes['color']
    return result

#extracts poll data from an XML string and converts to a DataFrame
def rcp_poll_data(xml):
    dom = web.Element(xml)
    result = {}

    # extract dates
    series = dom.by_tag('series')
    date_value = series[0].by_tag('value')
    date = []
    for d in date_value:
        date.append(pd.to_datetime(d.content))
    result['date'] = date
    
    #extract result data and titles
    graphs_tag = dom.by_tag('graphs')
    graph_tags = graphs_tag[0].by_tag('graph')
    
    for graph in graph_tags:
        title = graph.attributes['title']
        values = []
        for value in graph.by_tag('value'):
            try:
                values.append(float(value.content))
            except ValueError:
                values.append(np.nan)
        result[title] = values

    result = pd.DataFrame(result)
    return result

#from CS109 plots RCP poll data given the poll id
def poll_plot(poll_id):
    xml = get_poll_xml(poll_id)
    data = rcp_poll_data(xml)
    colors = plot_colors(xml)

    #remove characters like apostrophes
    data = data.rename(columns = {c: _strip(c) for c in data.columns})

    	#normalize poll numbers so they add to 100%    
    norm = data[colors.keys()].sum(axis=1) / 100    
    for c in colors.keys():
		data[c] /= norm
    
    for label, color in colors.items():
		plt.plot(data.date, data[label], color=color, label=label)        
        
		plt.xticks(rotation=70)
		plt.legend(loc='best')
		plt.xlabel("Date")
		plt.ylabel("Normalized Poll Percentage")

"""
Finds and returns links to RCP races on a page like
http://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html
"""
def find_governor_races(html):
    dom = web.Element(html)
    option_tags = dom.by_tag('option')

    gov_links = []
    
    #iterate through option tags
    for op in option_tags:
        value = op.attributes['value']
        # only append governor links
        if re.search("2010/governor", value):
            gov_links.append(value)
    return gov_links

#finds race result from RCP url    
def race_result(url):
    html = requests.get(url).text
    
    dom = web.Element(html)
    result = {}
    
    #find tags unique to candidate names
    tr_tags = dom.by_tag('tr.omit')
    th_tags = tr_tags[0].by_tag('th')
    

    #extract candidate names
    candidate = []
    
    #add names to candidate list without additional chars
    for tags in th_tags[3:-1]:
        if re.search("\(", tags.content):    
            candidate.append(tags.content[:-4]) 
        else:
            candidate.append(tags.content)
    
    #find tags unique to final polling results
    td_tags = tr_tags[0].next.by_tag('td')
    
    # extract percentages
    percentage = []
    for tags in td_tags[3:-1]:
        percentage.append(float(tags.content))
    
    result = dict(zip(candidate, percentage))

    return result

#CS109
def id_from_url(url):
    """Given a URL, look up the RCP identifier number"""
    return url.split('-')[-1].split('.html')[0]

#CS109
def plot_race(url):
    """Make a plot summarizing a senate race
    
    Overplots the actual race results as dashed horizontal lines
    """
    id = id_from_url(url)
    xml = get_poll_xml(id)    
    colors = plot_colors(xml)

    if len(colors) == 0:
        return
    
    result = race_result(url)
    
    poll_plot(id)
    plt.xlabel("Date")
    plt.ylabel("Polling Percentage")
    for r in result:
        plt.axhline(result[r], color=colors[_strip(r)], alpha=0.6, ls='--')

#CS109
def party_from_color(color):
    if color in ['#0000CC', '#3B5998']:
        return 'democrat'
    if color in ['#FF0000', '#D30015']:
        return 'republican'
    return 'other'


def error_data(url):
    """
    Provided by CS109
    Given a Governor race URL, download the poll data and race result,
    and construct a DataFrame with the following columns:
    
    candidate: Name of the candidate
    forecast_length: Number of days before the election
    percentage: The percent of poll votes a candidate has.
                Normalized to that the canddidate percentages add to 100%
    error: Difference between percentage and actual race reulst
    party: Political party of the candidate
    
    The data are resampled as necessary, to provide one data point per day
    """
    
    id = id_from_url(url)
    xml = get_poll_xml(id)
    
    colors = plot_colors(xml)
    if len(colors) == 0:
        return pd.DataFrame()
    
    df = rcp_poll_data(xml)
    result = race_result(url)
    
    #remove non-letter characters from columns
    df = df.rename(columns={c: _strip(c) for c in df.columns})
    for k, v in result.items():
        result[_strip(k)] = v 
    
    candidates = [c for c in df.columns if c is not 'date']
        
    #turn into a timeseries...
    df.index = df.date
    
    #...so that we can resample at regular, daily intervals
    df = df.resample('D')
    df = df.dropna()
    
    #compute forecast length in days
    #(assuming that last forecast happens on the day of the election, for simplicity)
    forecast_length = (df.date.max() - df.date).values
    forecast_length = forecast_length / np.timedelta64(1, 'D')  # convert to number of days
    
    #compute forecast error
    errors = {}
    normalized = {}
    poll_lead = {}
    
    for c in candidates:
        #turn raw percentage into percentage of poll votes
        corr = df[c].values / df[candidates].sum(axis=1).values * 100.
        err = corr - result[_strip(c)]
        
        normalized[c] = corr
        errors[c] = err
        
    n = forecast_length.size
    
    result = {}
    result['percentage'] = np.hstack(normalized[c] for c in candidates)
    result['error'] = np.hstack(errors[c] for c in candidates)
    result['candidate'] = np.hstack(np.repeat(c, n) for c in candidates)
    result['party'] = np.hstack(np.repeat(party_from_color(colors[_strip(c)]), n) for c in candidates)
    result['forecast_length'] = np.hstack(forecast_length for _ in candidates)
    
    result = pd.DataFrame(result)
    return result

#collects error data for all races found on RCP url
def all_error_data():
	page = requests.get('http://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html').text.encode('ascii', 'ignore')
	races = find_governor_races(page)
   
    #must skip url for first link, because access denied
	df = error_data(races[1])
    #add error data for each governor race to a dataframe
	for url in races[2:]:
		new = error_data(url)
		df = pd.DataFrame(df.append(new, ignore_index=True))
	return df

#makes prediction of current governor races given previous polling data
def bootstrap_estimate(errors,n,url):
    #get current RCP polling average for candidate of interest
	current_data = race_result(url)
	candidate1_perc = current_data[current_data.keys()[0]]
	candidate2_perc = current_data[current_data.keys()[1]]
    
	cand_1_wins = 0
	for i in xrange(n):
        #randomly select a number from errors dataframe
		er = random.choice(errors.error)
    
        #estimate outcome of race by adding/subtracting error from current data
		estimate_1 = candidate1_perc + er
		estimate_2 = candidate2_perc - er
    
		if estimate_1 > estimate_2: 
			cand_1_wins += 1
        
	cand1_win_percentage = cand_1_wins/1000.0*100
    
	print "The likelihood that " + current_data.keys()[0] + " wins the election is: " + str(cand1_win_percentage) + "%"
	return cand1_win_percentage

#test functions
if __name__ == '__main__':
	errors = all_error_data()
	poll_plot(1044)
	plt.title("Obama Job Approval")

	page = requests.get('http://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html').text.encode('ascii', 'ignore')


	#plot of polling data leading up to election
	for race in find_governor_races(page):
		plot_race(race)
		plt.show()
	

	#histogram showing polling errors from governor races
	errors.error.hist(bins=50)
	plt.xlabel("Polling Error")
	plt.ylabel('N')

	#standard deviation of governor polling data
	std = np.std(errors.error)
	print "the standard deviation of the polling errors is: " + str(std)

	errors_week = []

	#shows standard deviation of errors within one week of the election
	errors_week = errors[errors.forecast_length < 7]
	std_week = np.std(errors_week.error)
	print "Standard deviation of polling errors within one week of polling: " + str(std_week)

	#shows standard deviation of errors within one month of the election
	errors_month = errors[errors.forecast_length < 30]
	std_month = np.std(errors_month.error)    
	print "Standard deviation of polling errors within one month of polling: " + str(std_month)

	#make prediction of current governor races using bootsrap strategy
	bootstrap_estimate(errors, 1000, 'http://www.realclearpolitics.com/epolls/2013/governor/va/virginia_governor_cuccinelli_vs_mcauliffe-3033.html')
	bootstrap_estimate(errors, 1000, 'http://www.realclearpolitics.com/epolls/2013/governor/nj/new_jersey_governor_christie_vs_buono-3411.html')