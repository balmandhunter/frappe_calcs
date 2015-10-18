import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import pylab as pl
from scipy import stats
from sklearn.learning_curve import learning_curve
from sklearn.metrics import make_scorer
from sklearn import cross_validation
from matplotlib.font_manager import FontProperties


def plot_params():
    size = 18
    a = plt.rc('xtick', labelsize = size)
    b = plt.rc('ytick', labelsize = size)
    return a, b, plt.gca(), size


def plot_tr_and_holdout(df, pod_num, ref_column, label, cutoff):
    plt.figure(facecolor='w', figsize = (15,20))
    a, b, axes, label_size = plot_params()
    plt.ylabel('Ozone Concentration (ppb)', size = label_size)
    plt.xlabel('Date', size = label_size)

    ax1 = plt.subplot(311)
    df[ref_column].plot(marker = '.',linestyle = '-', label = label)
    xlim = assign_pod_calibration_times(pod_num, 1)
    ax1.set_xlim(xlim)
    plt.axhline(y = cutoff, color = 'r', linestyle = '-.')

    ax2 = plt.subplot(312)
    df[ref_column].plot(marker = '.',linestyle = '-', label = 'Training Data')
    xlim = assign_pod_calibration_times(pod_num, 2)
    ax2.set_xlim(xlim)
    plt.axhline(y = cutoff, color = 'r', linestyle = '-.')

    ax3 = plt.subplot(313)
    df[ref_column].plot(marker = '.',linestyle = '-', label = 'Training Data')
    xlim = assign_pod_calibration_times(pod_num, 3)
    ax3.set_xlim(xlim)
    plt.axhline(y = cutoff, color = 'r', linestyle = '-.')


def plot_03_temp_rh(df, pod_num, ref_column, label, cutoff, chunk):
    plt.figure(facecolor='w', figsize = (15,20))
    a, b, axes, label_size = plot_params()
    plt.xlabel('Time', size = 18)

    ax1 = plt.subplot(411)
    df['e2v03'].plot(marker = '.',linestyle = '-', label = label)
    xlim = assign_pod_calibration_times(pod_num, chunk)
    plt.ylabel('MOx Sensor Ozone Signal ', size = label_size)
    ax1.set_xlim(xlim)
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = plt.subplot(412)
    df[ref_column].plot(marker = '.',linestyle = '-', label = label)
    plt.ylabel('Ozone Concentration (ppb)', size = label_size)
    ax2.set_xlim(xlim)
    ax2.set_ylim([0,80])
    ax2.axes.get_xaxis().set_visible(False)

    ax3 = plt.subplot(413)
    df['Temp'].plot(marker = '.',linestyle = '-', label = label)
    plt.ylabel('Scaled Temperature', size = label_size)
    ax3.set_xlim(xlim)
    ax3.axes.get_xaxis().set_visible(False)

    ax4 = plt.subplot(414)
    df['Rh'].plot(marker = '.',linestyle = '-', label = label)
    plt.ylabel('Scaled Relative Humidity', size = label_size)
    ax4.set_xlim(xlim)


def plot_03_temp_rh_no_scale(df, pod_num, ref_column, label, cutoff, chunk):
    plt.figure(facecolor='w', figsize = (5,20))
    a, b, axes, label_size = plot_params()
    plt.xlabel('Time', size = 18)

    ax1 = plt.subplot(411)
    df['e2v03'].plot(marker = '.',linestyle = '-', label = label)
    xlim = assign_pod_calibration_times(pod_num, chunk)
    plt.ylabel('MOx Sensor Ozone Signal ', size = label_size)
    ax1.set_xlim(xlim)
    ax1.set_ylim([600,620])
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = plt.subplot(412)
    df[ref_column].plot(marker = '.',linestyle = '-', label = label)
    plt.ylabel('Ozone Concentration (ppb)', size = label_size)
    ax2.set_xlim(xlim)
    ax2.set_ylim([20,80])
    ax2.axes.get_xaxis().set_visible(False)

    ax3 = plt.subplot(413)
    df['Temp'].plot(marker = '.',linestyle = '-', label = label)
    plt.ylabel('Temperature (C)', size = label_size)
    ax3.set_xlim(xlim)
    ax3.set_ylim([45,47])
    ax3.axes.get_xaxis().set_visible(False)

    ax4 = plt.subplot(414)
    df['Rh'].plot(marker = '.',linestyle = '-', label = label)
    plt.ylabel('Relative Humidity', size = label_size)
    ax4.set_xlim(xlim)
    ax4.set_ylim([16,25])



def plot_hist(values, other, title):
    plt.figure(figsize = (10,5), facecolor='w')
    a, b, axes, label_size = plot_params()
    h = sorted(values)
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))
    #pl.plot(h, fit, '-o')
    plt.title(title, size = label_size)
    abs_min_dec = min(min(values), min(other))
    abs_max_dec = max(max(values), max(other))
    abs_min = myround(abs_min_dec, 5)
    abs_max = myround(abs_max_dec, 5)
    pl.hist(h, normed = True, bins=np.arange(abs_min-10,abs_max+10, 5))
    axes.set_xlim([-20, 90])
    axes.set_xlabel('Ozone Concentration (ppb)', size = label_size)
    pl.show()


def rmse_scoring_function(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))


def plot_learning_curve(estimator, title, X, y, ylimit, days_tr, train_sizes):
    plt.figure(facecolor='w', figsize = (8,8), frameon = "True")
    a, b, axes, label_size = plot_params()
    plt.title(title, size = label_size)
    if ylimit is not None:
        axes.set_ylim(ylimit)
    plt.xlabel("Training Samples", size = label_size)
    plt.ylabel("Root Mean Squared Error", size = label_size)
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y,
        cv = 5, train_sizes = train_sizes, scoring = make_scorer(rmse_scoring_function, greater_is_better = False))
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = -np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    #plt.grid(b=True, which='major', color='#696969', linestyle=':')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
        alpha=0.1, color="r")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std,
        alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training RMSE")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation RMSE")

    leg = plt.legend(loc="best", fontsize = label_size, frameon = 'True')
    leg.get_frame().set_facecolor('w')
    #fig.savefig('learning_curve.png', bbox_inches= 'tight')
    return plt


def fitted_vs_ref_plot(df, i, ref_column):
    plt.figure(facecolor='w', figsize = (8,8))
    a, b, axes, label_size = plot_params()
    plt.plot(df.ref_fit, df.O3_fit, linestyle = '', marker = '.', alpha = 0.3)
    plt.xlabel('Reference Ozone Concentration (ppb)', size = label_size)
    plt.ylabel('Predicted Cross-Val. Ozone Conc. (ppb)', size = label_size)
    plt.plot([1, df.ref_fit.max()], [1,df.ref_fit.max()])
    axes.set_ylim([-20,100])
    if i != 0:
        plt.title('Number of features = ' + str(i), size = label_size)


#Define a function that assigns time chunks to each pod for plotting
def assign_pod_calibration_times(pod_num, time_chunk):
    if time_chunk == 1:
        if pod_num == 'F3' or pod_num == 'F4' or pod_num == 'F5' or pod_num == 'F6'  or pod_num == 'F7' or pod_num == 'D0':
            xlim = ['2014-07-11 00:00:00', '2014-07-13 00:00:00']
        elif pod_num == 'F8':
            xlim = ['2014-07-11 00:00:00', '2014-7-12 00:00:00']
        elif pod_num == 'D4' or pod_num == 'D6'  or pod_num == 'N4' or pod_num == 'N7' or pod_num == 'N8':
            xlim = ['2014-07-13 00:00:00', '2014-7-15 00:00:00']
        elif pod_num == 'D8':
            xlim = ['2014-07-13 00:00:00', '2014-7-15 00:00:00']
        elif pod_num == 'N3' or pod_num == 'N5' or pod_num == 'D3':
            xlim = ['2014-07-8 00:00:00', '2014-7-11 00:00:00']
        elif pod_num == 'F9':
            xlim = ['2014-08-15 00:00:00', '2014-8-22 00:00:00']
        elif pod_num == 'cham_2':
            xlim = ['2014-07-25 00:00:00', '2014-8-2 00:00:00']
    elif time_chunk == 2:
        if pod_num == 'D0' or pod_num == 'F8':
            xlim = ['2014-08-30 00:00:00', '2014-09-1 00:00:00']
        elif pod_num == 'D4' or pod_num == 'F4':
            xlim = ['2014-08-15 00:00:00', '2014-08-17 00:00:00']
        elif pod_num == 'D0':
            xlim = ['2014-08-29 00:00:00', '2014-09-1 00:00:00']
        elif (pod_num == 'D6' or pod_num == 'F3' or pod_num == 'D8' or pod_num == 'F5' or
            pod_num == 'F6' or pod_num == 'N8'):
            xlim = ['2014-08-21 00:00:00', '2014-08-25 00:00:00']
        elif pod_num == 'F7' or pod_num == 'N4':
            xlim = ['2014-08-15 00:00:00', '2014-08-17 00:00:00']
        elif pod_num == 'N3':
            xlim = ['2014-08-14 00:00:00', '2014-08-18 00:00:00']
        elif pod_num == 'D4' or pod_num == 'N5':
            xlim = ['2014-08-29 00:00:00', '2014-09-4 00:00:00']
        elif pod_num == 'N7':
            xlim = ['2014-08-16 00:00:00', '2014-08-18 00:00:00']
        elif pod_num == 'D3':
            xlim = ['2014-08-21 00:00:00', '2014-08-25 00:00:00']
        elif pod_num == 'F9':
            xlim = ['2014-08-22 00:00:00', '2014-8-27 00:00:00']
        elif pod_num == 'cham_2':
            xlim = ['2014-08-2 00:00:00', '2014-8-9 00:00:00']
    elif time_chunk == 3:
        if pod_num == 'D3':
            xlim = ['2014-08-26 00:00:00', '2014-08-30 00:00:00']
        elif pod_num == 'D0' or pod_num == 'F8':
            xlim = ['2014-09-1 00:00:00', '2014-09-4 00:00:00']
        elif pod_num == 'D4' or pod_num == 'F4':
            xlim = ['2014-08-18 00:00:00', '2014-08-21 00:00:00']
        elif pod_num == 'D0':
            xlim = ['2014-09-2 00:00:00', '2014-09-4 00:00:00']
        elif (pod_num == 'D6' or pod_num == 'F3' or pod_num == 'D8' or pod_num == 'F5' or
            pod_num == 'F6' or pod_num == 'N8'):
            xlim = ['2014-08-26 00:00:00', '2014-08-30 00:00:00']
        elif pod_num == 'F7' or pod_num == 'N4':
            xlim = ['2014-08-17 00:00:00', '2014-08-21 00:00:00']
        elif pod_num == 'N3':
            xlim = ['2014-08-19 00:00:00', '2014-08-21 00:00:00']
        elif pod_num == 'D4' or pod_num == 'N5':
            xlim = ['2014-08-29 00:00:00', '2014-09-4 00:00:00']
        elif pod_num == 'N7':
            xlim = ['2014-08-18 00:00:00', '2014-08-22 00:00:00']
        elif pod_num == 'F9':
            xlim = ['2014-08-27 00:00:00', '2014-9-4 00:00:00']
        elif pod_num == 'cham_2':
            xlim = ['2014-08-9 00:00:00', '2014-8-16 00:00:00']
    elif time_chunk == 4:
        if pod_num == 'cham_2':
            xlim = ['2014-07-27 17:00:00', '2014-7-28 1:00:00']
    elif time_chunk == 5:
        if pod_num == 'cham_2':
            xlim = ['2014-08-3 11:00:00', '2014-8-3 18:00:00']
    elif time_chunk == 6:
        if pod_num == 'cham_2':
            xlim = ['2014-08-4 10:00:00', '2014-8-4 19:00:00']

    return xlim


def plot_fitted_and_ref_vs_time(df, pod_num, time_chunk, ref_column):
    plt.figure(facecolor='w', figsize = (15,10))
    a, b, axes, label_size = plot_params()
    df.ref_fit.plot(marker = '.',linestyle = '-', label = 'Reference Data')
    if time_chunk != 0:
        xlim = assign_pod_calibration_times(pod_num, time_chunk)
        df.O3_fit.plot(marker = '.',linestyle = '-', xlim = xlim, label = 'Predicted Data')
    else:
        df.O3_fit.plot(marker = '.',linestyle = '-', label = 'Predicted Data')
    axes.set_ylim([-10,90])
    plt.legend(fontsize = label_size, loc = "best")
    plt.ylabel('Ozone Concentration (ppb)', size = label_size)
    plt.xlabel('Date', size = label_size)


def plot_fitted_and_ref_vs_time_no_lines(df, pod_num, time_chunk, ref_column):
    plt.figure(facecolor='w', figsize = (15,10))
    a, b, axes, label_size = plot_params()
    df.ref_fit.plot(marker = '.',linestyle = '--', label = 'Reference')
    if time_chunk != 0:
        xlim = assign_pod_calibration_times(pod_num, time_chunk)
        df.O3_fit.plot(marker = '.',linestyle = '', color='r', xlim = xlim, label = 'Predicted')
    else:
        df.O3_fit.plot(marker = '.',linestyle = '', color='r', label = 'Predicted')
    axes.set_ylim([-10,90])
    plt.legend(fontsize = label_size, loc = "best")
    plt.ylabel('Ozone Concentration (ppb)', size = label_size)
    plt.xlabel('Date', size = label_size)


def plot_field(df_best, df_base, df_tower, time_chunk):
    plt.figure(facecolor='w', figsize = (20,5))
    a, b, axes, label_size = plot_params()
    if time_chunk == 1:
        df_best.O3_fit.plot(marker = '.',linestyle = ' ', label = 'Best Features', xlim = ['2014-07-22 00:00:00', '2014-8-1 00:00:00'], alpha = 0.5, color = '#fc8d62')
        df_base.O3_fit.plot(marker = '.',linestyle = ' ', label = 'Base Features', xlim = ['2014-07-22 00:00:00', '2014-8-1 00:00:00'], alpha = 0.5, color = '#66c2a5')
        df_tower.ozone.plot(marker = '.',linestyle = ' ', label = '2b data', xlim = ['2014-07-22 00:00:00', '2014-8-1 00:00:00'], alpha = 0.5, color = '#8da0cb')
        plt.axhline(y = 75, color = 'r', linestyle = '-.')


    elif time_chunk == 2:
        df_best.O3_fit.plot(marker = '.',linestyle = ' ', label = 'Best Features', xlim = ['2014-08-1 00:00:00', '2014-8-10 00:00:00'], alpha = 0.5, color = '#fc8d62')
        df_base.O3_fit.plot(marker = '.',linestyle = ' ', label = 'Base Features', xlim = ['2014-08-1 00:00:00', '2014-8-10 00:00:00'], alpha = 0.5, color = '#66c2a5')
        df_tower.ozone.plot(marker = '.',linestyle = ' ', label = '2b data', xlim = ['2014-08-1 00:00:00', '2014-8-10 00:00:00'], alpha = 0.5, color = '#8da0cb')
        plt.axhline(y = 75, color = 'r', linestyle = '-.')

    elif time_chunk == 3:
        df_best.O3_fit.plot(marker = '.',linestyle = ' ', label = 'Best Features', xlim = ['2014-8-10 00:00:00', '2014-8-20 00:00:00'], alpha = 0.5, color = '#fc8d62')
        df_base.O3_fit.plot(marker = '.',linestyle = ' ', label = 'Base Features', xlim = ['2014-8-10 00:00:00', '2014-8-20 00:00:00'], alpha = 0.5, color = '#66c2a5')
        df_tower.ozone.plot(marker = '.',linestyle = ' ', label = '2b data', xlim = ['2014-8-10 00:00:00', '2014-8-20 00:00:00'], alpha = 0.5, color = '#8da0cb')
        plt.axhline(y = 75, color = 'r', linestyle = '-.')

    axes.set_ylim([0,90])
    plt.ylabel('Ozone Concentration (ppb)', size = label_size)
    plt.xlabel('Date', size = label_size)


def myround(x, base):
    return int(base * round(float(x)/base))


def plot_error_vs_features(score, RMSE, ylim, xlim):
    plt.figure(facecolor='w', figsize = (10,5))
    a, b, axes, label_size = plot_params()
    x = range(1, len(score)+1)
    plt.plot(x, score, marker = '.', markersize = 20, label='Custom Score')
    plt.plot(x, RMSE, marker = '.', markersize = 20, label='RMSE')
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    plt.xlabel('Number of Features', size = label_size)
    plt.ylabel('Error', size = label_size)
    #plt.grid(b=True, which='major', color='g', linestyle='-.')
    plt.legend(fontsize = label_size, loc = "best")


def plot_resid_vs_conc(df, ref_column):
    #find the residuals
    resid = df.ref_fit - df.O3_fit
    #plot the residuals to check for non-linearity of response predictor
    plt.figure(facecolor='w', figsize = (15,5))
    a, b, axes, label_size = plot_params()
    plt.plot(df.O3_fit, resid, linestyle = '',marker = '.',alpha = 0.4)
    plt.plot([-40,70],[0,0], linestyle = ' ', marker = '.')
    axes.set_ylim([-80,80])
    axes.set_xlim([-20,100])
    plt.xlabel('Predicted Ozone Concentration', size = label_size)
    plt.ylabel('Residuals', size = label_size)
    return resid


def plot_resid_vs_time(resid, pod_num, time_chunk):
    plt.figure(facecolor='w', figsize = (15,5))
    a, b, axes, label_size = plot_params()
    xlim = assign_pod_calibration_times(pod_num, time_chunk)
    resid.plot(linestyle = '',marker = '.', xlim = xlim)
    #plt.plot([0,0],[70,0])
    plt.xlabel('Fitted O3 Conc.', size = label_size)
    plt.ylabel('Residuals', size = label_size)


def plot_lambda(lambda_ridge, coefs, mean_score_lambda):
    #plot the coefficients
    plt.figure(facecolor='w', figsize = (10,5))
    a, b, axes, label_size = plot_params()
    #axes.set_ylim(-50, 50)
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

    ax.plot(lambda_ridge, coefs)
    ax.set_xscale('log')
    plt.xlabel('lambda', size = label_size)
    plt.ylabel('weights', size = label_size)
    plt.title('coefficients as a function of the regularization', size = label_size)
    plt.show()

    #plot the results
    plt.figure(facecolor='w', figsize = (10,5))
    plt.plot(lambda_ridge, mean_score_lambda)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel('lambda', size= label_size)
    plt.ylabel('Custom Score', size = label_size)


def plot_importance(rfr,forest, features):
    importances = rfr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    print std
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(10):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])),features[indices[f]]
    #Plot the feature importances of the forest
    plt.figure(figsize=(15,5))
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices], color="r", align="center")
    #, yerr = std[indices]
    plt.xticks(range(len(features)), indices)
    plt.xlim([-1, len(features)])
    plt.show()


def plot_temp_and_rh_for_each_day(df_T, df_H):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.title('Temp and Rh Data', fontsize = 30)
    ax.plot(df_H['Temp'],  color="r", marker = '.', linestyle = '--', label = 'reference')
    ax.set_xlabel('Time', fontsize = 18)
    ax.set_ylabel('Temperature (as % of maximum)', fontsize = 18)
    ax.legend()
    ax2 = ax.twinx()
    ax2.set_ylabel('Rel. Hum. (as % of maximum)', fontsize = 18)
    ax2.legend(loc = 0)
    plt.plot((df_H['Rh']), marker = '.', linestyle = '--', label = 'pod')
    plt.show()


def plot_ref_and_pod_ozone_for_each_day(df_T, df_H):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.title('Pod and Reference Ozone Data', fontsize = 30)
    ax.plot(df_H['O3_ppb'], color="r", marker = '.', linestyle = '--', label = 'reference')
    ax.set_xlabel('Time', fontsize = 18)
    ax.set_ylabel('Reference Ozone', fontsize = 18)
    ax.legend()

    df_H['ones'] = 1
    df_H['inverse_o3'] = df_H['ones'].div(df_H['e2v03'], axis='index')
    ax2 = ax.twinx()
    ax2.set_ylabel('Pod Ozone (1/mV)', fontsize = 18)
    ax2.legend(loc = 0)
    plt.plot((df_H['inverse_o3']), marker = '.', linestyle = '--', label = 'pod')
    plt.show()


def plot_param_select_MSE(MSE_CV_per_day, i, j):
    fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    imgplot = plt.imshow(MSE_CV_per_day)
    imgplot.set_cmap('hot')
    #imgplot.set_clim(60,71)
    ax.set_aspect('equal')

    plt.colorbar(orientation='vertical')
    plt.show()
    plt.xlabel('Maximum Tree Depth')
    plt.ylabel('Maximum Features at Each Split')

    i,j = np.where(MSE_CV_per_day == MSE_CV_per_day.min())
    print 'Max features = ' + str(i)
    print 'Max depth = ' + str(j)
    print 'MSE for the holdout data = ' + str(min_MSE_CV)


def plot_fitted_and_ref_ozone_for_each_day(fitted_data, df_H):
    plt.figure(figsize=(15,5))
    plt.title('Fitted and Ref. Ozone Data', fontsize = 30)
    fitted_data.plot(color="r", marker = '.', label = 'fitted')
    plt.xlabel('Time', fontsize = 18)
    plt.ylabel('Ozone (ppb)', fontsize = 18)
    plt.legend()
    df_H['O3_ppb'].plot(label = 'reference')
    plt.show()


def plot_daily_mse_and_features_for_day(MSE_H, day_date,feat_to_compare, title, sec_axis_label):
    from matplotlib import rc
    rc('mathtext', default='regular')
    indices = day_date
    #Plot the feature importances of the forest
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.title(title, fontsize = 30)
    ax.bar(range(len(day_date)), MSE_H,  color="r", align="center")
    plt.xticks(range(len(day_date)), indices)
    plt.xlim([-1, len(day_date)])
    ax.set_xlabel('Date', fontsize = 18)
    ax.set_ylabel('MSE (ppb)', fontsize = 18)
    ax2 = ax.twinx()
    ax2.set_ylabel(sec_axis_label, fontsize = 18)
    plt.plot(range(len(day_date)), feat_to_compare, marker = 'o', linestyle = '--')
    plt.show()


if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
