import numpy as np
import pandas as pd
from zipfile import ZipFile
import os
import re
import numpy.typing as npt
import numexpr as ne
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore as perc_sc
from typing import Optional
from statsmodels.stats.multitest import multipletests, fdrcorrection


def get_zipfile_names(directory: str) -> list[str]:
    list_zipfiles = []
    for filename in os.listdir(directory):
        str_filename = os.path.join(directory, filename)
        if str_filename.endswith('.zip'):
            list_zipfiles.append(str_filename)
    return(list_zipfiles)


def get_scenario_names(zipname: str) -> list[str]:
    with ZipFile(zipname, 'r') as zipos:
        list_names_csv = [name for name in zipos.namelist() if ((name.endswith('.csv')) & ('Challenge' in name))]
        list_names_csv_slashless = [name if '/' not in name else name.rsplit('/', 1)[1] for name in list_names_csv]
    return(list_names_csv_slashless)


def create_df_from_zipcsvs(list_zipfiles: list[str], names_of_scenarios: list[str]) -> pd.DataFrame:
    list_players = []
    list_scenarios = []
    list_strings_of_shots = []
    list_hits = []
    list_misses = []
    list_accuracy = []
    list_n_shots = []
    list_dates = []
    list_times = []
    list_lengths = []
    for player, zipname in enumerate(list_zipfiles, 1):
        with ZipFile(zipname, 'r') as zipos:
                list_names_csv = [name for name in zipos.namelist() if ((name.endswith('.csv')) & ('Challenge' in name))]
                list_names_csv_slashless = [name if '/' not in name else name.rsplit('/', 1)[1] for name in list_names_csv]
                list_csv = [filename for filename, name in zip(list_names_csv, list_names_csv_slashless) if re.search('(.*)(?= - Challenge)', name)[0] in names_of_scenarios]
                for filename in list_csv:
                    scenario = re.search('(.*)(?= - Challenge)', filename)[0] if '/' not in filename else re.search('(.*)(?= - Challenge)', filename.rsplit('/', 1)[1])[0]
                    date = re.search('(?<=Challenge - )(.*)(?= Stats.csv)', filename)[0][:10]
                    time = re.search('(?<=Challenge - )(.*)(?= Stats.csv)', filename)[0][11:]
                    with zipos.open(filename) as stats:
                        statos = pd.read_csv(stats, header = 0, on_bad_lines = 'skip').dropna()
                    shots = statos['Shots'].to_numpy()
                    n_shots = np.sum(shots)
                    if n_shots > 30:
                        timestamp = pd.to_datetime(statos['Timestamp'].map(lambda x: x[:-4]), format = '%H:%M:%S')
                        length = min([30*n for n in range(1, 5)], key = lambda x: abs(x-(timestamp[len(timestamp)-1] - timestamp[1]).seconds))
                        shots2 = np.cumsum(shots, dtype = np.uint32)
                        str_shots = np.zeros(shots2[-1])
                        str_shots[shots2-1] = 1
                        hits = shots.size
                        misses = n_shots - hits
                        accuracy = hits / n_shots
                        list_players.append(player)
                        list_scenarios.append(scenario)
                        list_strings_of_shots.append(str_shots)
                        list_hits.append(hits)
                        list_misses.append(misses)
                        list_accuracy.append(accuracy)
                        list_n_shots.append(n_shots)
                        list_dates.append(date)
                        list_times.append(time)
                        list_lengths.append(length)

    df = pd.DataFrame(dict(Player = list_players, Scenario = list_scenarios, Length = list_lengths, String_of_shots = list_strings_of_shots, Number_of_shots = list_n_shots, Hits = list_hits, Misses = list_misses, Accuracy = list_accuracy, Date = list_dates, Time = list_times))
    df['Player'] = pd.Categorical(df['Player'])
    df['Scenario'] = pd.Categorical(df['Scenario'])
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y.%m.%d')
    df['Time'] = pd.to_datetime(df['Time'], format = '%H.%M.%S')
    df = df.sort_values(by = ['Player', 'Scenario', 'Date', 'Time'])
    df['Attempt'] = df.reset_index().groupby(['Player', 'Scenario'])['index'].rank()
    df['Attempt_that_day'] = df.reset_index().groupby(['Player', 'Scenario', 'Date'])['index'].rank()
    df = df.groupby(['Player', 'Scenario']).apply(lambda df: df.assign(Accuracy_per_player_scenario = lambda df: df['Hits'].sum() / df['Number_of_shots'].sum()))
    df = df.reset_index(drop=True)
    df = df.groupby(['Scenario']).apply(lambda df: df.assign(Accuracy_per_scenario = lambda df: df['Hits'].sum() / df['Number_of_shots'].sum()))
    return(df)


def get_P_hat_and_p(shots: npt.ArrayLike, reps: int, k: int) -> np.ndarray:
    arr = np.concatenate(([0], shots.flatten(), [0]))
    absdiff = np.abs(np.diff(arr))
    ranges = np.flatnonzero(absdiff == 1).reshape(-1, 2)

    if shots.ndim > 1:
        len_array = shots.shape[1]
    else:
        len_array = shots.size

    if reps > 1:
        len_mults = np.floor(ranges[:, 1] / len_array)*len_array
        cuts, = np.where(np.logical_and(ranges[:,0] < len_mults, len_mults < ranges[:,1]))
        left = np.c_[ranges[cuts][:,0], len_mults[cuts]]
        right = np.c_[len_mults[cuts], ranges[cuts][:,1]]
        cuts_size = np.arange(cuts.size)
        ranges = np.delete(ranges, cuts, axis = 0)
        ranges = np.insert(ranges, cuts - cuts_size, left, axis = 0)
        ranges = np.insert(ranges, cuts + 1, right, axis = 0)

    k_cons = ranges[:,1]-ranges[:,0]-k+1
    k_cons[np.nonzero(k_cons < 0)] = 0
    n_hits_all = np.where(k_cons > 0, k_cons-1, 0)
    ends = np.arange(1, reps+1)*len_array
    is_not_last = np.where(np.isin(ranges[:,1], ends), 0, 1)
    n_misses_all = np.where(k_cons == 0, 0, is_not_last)
    bounds_array = np.floor(ranges[:, 0] / len_array)*len_array
    _, bound_ixs = np.unique(bounds_array, return_index = True, axis = 0)
    n_hits = np.add.reduceat(n_hits_all, bound_ixs)
    n_misses = np.add.reduceat(n_misses_all, bound_ixs)
    P_hat = ne.evaluate('n_hits / (n_hits + n_misses)')
    if shots.ndim > 1:
        p = ne.evaluate('sum((1 / len_array) * shots, axis=1)')
    else:
        p = ne.evaluate('sum((1 / len_array) * shots)')
    P_hat_p = P_hat - p
    return(P_hat, p, P_hat_p)


def get_D_hat_P_hat_p(shots: npt.ArrayLike, reps: int, k: int) -> np.ndarray:
    arr = shots.copy()
    P_hat_1, p, P_hat_p = get_P_hat_and_p(shots = arr, reps = reps, k = k)
    arr = 1-arr
    P_hat_0, _, _ = get_P_hat_and_p(shots = arr, reps = reps, k = k)
    D_hat = ne.evaluate('P_hat_1 - P_hat_0')
    D_hat = D_hat[np.isfinite(D_hat)]
    P_hat_p = P_hat_p[np.isfinite(P_hat_p)]
    return(D_hat, P_hat_p, p)


def get_D_hat_P_hat_p_dists(shots: npt.ArrayLike, k: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array_of_arrays = np.tile(shots, (reps,1))
    array_of_arrays = rng.permuted(array_of_arrays, axis = 1)
    D_hat, P_hat_p, p = get_D_hat_P_hat_p(shots = array_of_arrays, reps = reps, k = k)
    return(D_hat, P_hat_p, p)


def get_P_hat_p_dists(shots: npt.ArrayLike, k: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = shots.copy()
    array_of_arrays = np.tile(arr, (reps,1))
    array_of_arrays = rng.permuted(array_of_arrays, axis = 1)
    _, _, P_hat_p = get_P_hat_and_p(shots = array_of_arrays, reps = reps, k = k)
    return(P_hat_p)


def plot_stat_dist(stat_distribution: npt.ArrayLike, stat: float, percentiles: list[float], bins: int) -> plt:
    _, ax = plt.subplots()
    _, bins, patches = ax.hist(stat_distribution, bins = bins, edgecolor = 'white', linewidth = 1)

    t_ks_perc = np.percentile(stat_distribution, percentiles)
    bin_95 = np.argmax(bins >= t_ks_perc[0])
    bin_99 = np.argmax(bins >= t_ks_perc[1])

    for i in range(0, bin_95):
        patches[i].set_facecolor('#9CC3FF')
    for i in range(bin_95, bin_99):
        patches[i].set_facecolor('#FFCF9C')
    for i in range(bin_99, len(patches)):
        patches[i].set_facecolor('#FF7878')

    ax.axvline(x = stat, color = '#599750', lw = 2) # napsat hodnotu vedle tý čáry
    return(plt, ax)


def get_p_values(shots: npt.ArrayLike, reps: int, k : int) -> np.ndarray:
    D_hat, P_hat_p, _ = get_D_hat_P_hat_p(shots = shots, reps = 1, k = k)
    D_hat_dist, P_hat_p_dist, _ = get_D_hat_P_hat_p_dists(shots = shots, k = k, reps = reps, seed = 42)
    D_had_al = perc_sc(D_hat_dist, D_hat, kind = 'rank')
    P_hat_p_al = perc_sc(P_hat_p_dist, P_hat_p, kind = 'rank')
    D_had_p_value = ne.evaluate('(100-D_had_al)/100')
    P_hat_p_p_value = ne.evaluate('(100-P_hat_p_al)/100')
    return(D_had_p_value, P_hat_p_p_value)


def get_p_values_2(shots: npt.ArrayLike, reps: int, k : int) -> np.ndarray:
    _, _, P_hat_p = get_P_hat_and_p(shots = shots, reps = reps, k = k)
    P_hat_p_dist = get_P_hat_p_dists(shots = shots, k = k, reps = reps, seed = 42)
    P_hat_p_al = perc_sc(P_hat_p_dist, P_hat_p, kind = 'rank')
    P_hat_p_p_value = ne.evaluate('(100-P_hat_p_al)/100')
    return(P_hat_p_p_value)


def get_n_rejets(p_values: npt.ArrayLike, alpha: int, sidak: Optional[bool] = False) -> int:
    if sidak:
        fwer_p = multipletests(p_values, alpha = alpha, method = 'holm-sidak')[0]
        hot_hands_95 = np.flatnonzero(fwer_p).size
    else:
        hot_hands_95 = np.flatnonzero(p_values < alpha).size
    return(hot_hands_95)


def get_FDR(p_values: npt.ArrayLike, alpha: int) -> int:
    FDR_p = fdrcorrection(p_values, alpha = alpha, method = 'indep')[0]
    hot_hands_95 = np.flatnonzero(FDR_p).size
    return(hot_hands_95)


def entropy(shots: npt.ArrayLike):
    arr = shots.copy()
    a = np.concatenate(([1],arr,[1]))
    a_ixs = np.flatnonzero(a == 1)
    x_i = np.diff(a_ixs)
    H_s = -np.sum(x_i*np.log(x_i))
    return H_s


def entropy_dist(shots: npt.ArrayLike, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = shots.copy()
    array_of_arrays = np.tile(arr, (reps,1))
    array_of_arrays = rng.permuted(array_of_arrays, axis=1)
    a = np.column_stack((np.array([1]*array_of_arrays.shape[0]),array_of_arrays,np.array([1]*array_of_arrays.shape[0])))
    a_ixs = np.flatnonzero(a == 1).reshape(reps, -1)
    x_i = np.diff(a_ixs)
    H_s_dist = -np.sum(x_i*np.log(x_i), axis = 1)
    return H_s_dist


def entropy_pval(stat: int, dist: npt.ArrayLike) -> np.ndarray:
    stat_ = -stat.copy()
    dist_ = -dist.copy()
    pperc = perc_sc(dist_, stat_, kind = 'rank')
    pval = ne.evaluate('(100-pperc)/100')
    return(pval)


def plot_ent_dist(stat_distribution: npt.ArrayLike, stat: float, percentiles: list[float], bins: int) -> plt:
    _, ax = plt.subplots()
    _, bins, patches = ax.hist(stat_distribution, bins = bins, edgecolor = 'white', linewidth = 1)

    t_ks_perc = np.percentile(stat_distribution, percentiles)
    bin_95 = np.argmax(bins >= t_ks_perc[0])
    bin_99 = np.argmax(bins >= t_ks_perc[1])

    for i in range(0, bin_99):
        patches[i].set_facecolor('#FF7878')
    for i in range(bin_99, bin_95):
        patches[i].set_facecolor('#FFCF9C')
    for i in range(bin_95, len(patches)):
        patches[i].set_facecolor('#9CC3FF')

    ax.axvline(x = stat, color = '#599750', lw = 3) # napsat hodnotu vedle tý čáry
    return(plt, ax)


def weighted_entropy(shots: npt.ArrayLike):
    arr = shots.copy()
    a = np.concatenate(([1],arr,[1]))
    a_ixs = np.flatnonzero(a == 1)
    N = np.size(arr)
    n = np.sum(arr)
    x_i = np.diff(a_ixs)/(N+1)
    H_s = -np.sum(x_i*np.log(x_i))/np.log(n+1)
    return H_s


def weighted_entropy_dist(shots: npt.ArrayLike, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = shots.copy()
    array_of_arrays = np.tile(arr, (reps, 1))
    array_of_arrays = rng.permuted(array_of_arrays, axis=1)
    a = np.column_stack((np.array([1]*array_of_arrays.shape[0]), array_of_arrays, np.array([1]*array_of_arrays.shape[0])))
    a_ixs = np.flatnonzero(a == 1).reshape(reps, -1)
    N = np.size(arr)
    n = np.sum(arr)
    x_i = np.diff(a_ixs)/(N+1)
    H_s_dist = -np.sum(x_i*np.log(x_i), axis = 1)/np.log(n+1)
    return H_s_dist