# Get log files for each bird in list, then send those log files via email to specified address
import requests
import pandas as pd
import numpy as np
import click

import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import sys
#sys.path.append("/mnt/cube/tsainbur/glab-common-py")

import glob
import pandas as pd
import os
import datetime as dt
#import seaborn as sns
import scipy.ndimage
import numpy as np
data_path_zog = '/Users/annie1/Downloads/behav-analysis-python3_tim/Sample Behave Data'
data_path_vogel = '/Users/annie1/Downloads/behav-analysis-python3_tim/Sample Behave Data'

import sys
sys.path.append( '/Users/annie1/Documents/GitHub/behav-analysis/behav')
import plotting, utils, loading

def generate_bird_plots(subjects,fig_spot,data_path_zog):
    behav_data_zog = loading.load_data_pandas(subjects,data_path_zog)
    #behav_data_vogel= loading.load_data_pandas(subjects,data_path_vogel)
    behav_data = behav_data_zog.copy()
    #behav_data.update(behav_data_vogel)
    figs_list = {'cal':[], 'acc_bias':[], 'daily_acc':[]}
    for subj,data in behav_data.items():
        pc_fig = plotting.plot_filtered_performance_calendar(subj,data,num_days=20, return_fig=True)
        figs_list['cal'].append(''.join([fig_spot,'/performance_cal_',str(subj),'.png']))
        pc_fig.savefig(figs_list['cal'][-1])

        ci_acc_fig =  plotting.plot_ci_accuracy(subj, data, return_fig=True)
        figs_list['acc_bias'].append(''.join([fig_spot,'/ci_acc_',str(subj),'.png']))
        ci_acc_fig.savefig(figs_list['acc_bias'][-1])

        daily_acc_fig = plotting.plot_daily_accuracy(subj, data, x_axis='trial_num', return_fig=True)
        figs_list['daily_acc'].append(''.join([fig_spot,'/daily_acc_',str(subj),'.png']))
        daily_acc_fig.savefig(figs_list['daily_acc'][-1])
    return figs_list

def send_mail_local(send_from, send_to, subject,images=None,server='gmail-smtp-in.l.google.com:25',text='This is the alternative plain text message.'):
    
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    
    msgAlternative = MIMEMultipart('alternative')
    msgText = MIMEText(text)
    msgAlternative.attach(msgText)
    
    msg.attach(msgAlternative)
    
    html_attach_list = ['Today\'s bird results: \n<br>']
    
    for i in range(len(images)) or []:
        fp = open(images[i], 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()
        msgImage.add_header('Content-ID', ''.join(['<image',str(i),'>']))
        msg.attach(msgImage)
        html_attach_list.append(''.join(['<img width=900 src="cid:image',str(i),'"><br>']))
    
    msgText = MIMEText(''.join(html_attach_list), 'html')
    msgAlternative.attach(msgText)
    
    try:
        smtp = smtplib.SMTP("127.0.0.1:8888")
       # smtp.ehlo('gmail.com')
        #smtp.starttls()
        smtp.sendmail(send_from, send_to, msg.as_string())
        smtp.close()
        print("Email sent!")
    except:
        print("Error:unable to send email to {}".format(send_to))
        raise
    
    
# https://docs.google.com/spreadsheets/d/19BAwBWsDz9u5dFBjy5pOz3xLO4S-aU5N-_BQpiaxIgg/edit#gid=0
# Get the Google Sheets
sskey = '1fo2zuTxWuCfDpSV7krMmAo9A3pvfb6Ag7YKo3x5my1I'
spreadsheet_url = 'https://docs.google.com/spreadsheets/d/'+sskey+'/export?format=csv&id%22'
from io import StringIO

r = requests.get(spreadsheet_url)
data = r.content

bird_list = pd.DataFrame([i.split(',') for i in str(data.decode("utf-8") ).split('\r\n')])
bird_list.columns = bird_list.iloc[0]
bird_list = bird_list[1:]
fig_spot = '/Users/annie1/Documents/GitHub/behav-analysis/performance_figs'
@click.command()
@click.option('--email', help='Email sned from',prompt='what is your email')
@click.option('--fig_spot', prompt='Where do you want to put the temp pic to',
              help='The location to save the pictures')
@click.option('--data', prompt='Where is your data located',
              help='The location of your data')
def main(email,fig_spot,data):
    for email_add in list(bird_list.columns.values):
        print(np.array([i for i in bird_list[email_add] if type(i) == str]))
        figs_list = generate_bird_plots(np.array([i for i in bird_list[email_add] if type(i) == str]),fig_spot,data)
        send_from = email 
        send_to = [email_add]
        subject = 'Bird Results'
        send_mail_local(send_from, send_to, subject,np.concatenate([figs_list[i] for i in figs_list])[::-1])
if __name__ == '__main__':
    main()   
