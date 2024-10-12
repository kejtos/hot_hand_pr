import polars as pl
import numpy as np
from statsmodels.stats.multitest import multipletests, fdrcorrection
import numexpr as ne
from scipy.stats import percentileofscore as perc_sc
from datetime import datetime
from functions import *

_ = pl.Config.set_fmt_str_lengths(100)
_ = pl.Config.set_fmt_float('full')
# sns.set_palette('colorblind')
# COLORS = sns.color_palette()

TEAMS_PATH = 'final_data.csv'
PLAYER_PATH = 'final_dataset.csv'
PREDICTIONS_PATH = 'R\\R_output_files\\df_with_all_predictions.csv'

df = (
    pl.scan_csv(PREDICTIONS_PATH)
    .sort('player_id', 'date', 'match_id')
    .select('player_id', 'rating', 'time_gap_1', 'predicted_skill', 'predicted_skill_time_gap', 'predicted_skill_w_h', 'predicted_skill_w_log', 'predicted_skill_w_sr')
    .collect()
)

pl.read_csv(PREDICTIONS_PATH).n_unique('match_id')
pl.read_csv(PREDICTIONS_PATH).n_unique('event_id')

ALPHA = 0.05
REPS = 1_000_000
SEED = 42
K = 3

df_teams = (
    pl.scan_csv(TEAMS_PATH)
    .select(
        'Date', 'Event_id', 'Team_id', 'Player_id', 'Match_id', 'Klaassen_dif', 'Best-of', 'Team_star', 'Team_rank', 'Enemy_rank',
        'Elimination_match', 'Prizepool', 'For_money', 'Offline', 'Rounds_to_end', 'Log_rounds_to_end', 'Tier_A', 'Tier_B', 'Tier_C', 'Tier_S', 'Major'
    )
    .with_columns(
        pl.col('Date').cast(pl.Date),
        rank_diff=(pl.col('Team_rank') - pl.col('Enemy_rank')).abs()
    )
    .unique(['Match_id', 'Team_id'])
    .group_by('Match_id', 'Team_id')
    .agg(
        event_id=pl.first('Event_id'),
        date=pl.first('Date'),
        klaassen_dif=pl.first('Klaassen_dif'),
        best_of=pl.first('Best-of'),
        elimination_match=pl.first('Elimination_match'),
        prizepool=pl.first('Prizepool'),
        for_money=pl.first('For_money'),
        offline=pl.first('Offline'),
        log_rounds_to_end=pl.first('Log_rounds_to_end'),
        rounds_to_end=pl.first('Rounds_to_end'),
        tier_A=pl.first('Tier_A'),
        tier_B=pl.first('Tier_B'),
        tier_C=pl.first('Tier_C'),
        tier_S=pl.first('Tier_S'),
        major=pl.first('Major'),
        rank_diff=pl.first('rank_diff'),
    )
    .collect()
)

players = (
    pl.scan_csv(PLAYER_PATH)
    .filter(
        (pl.col('Side') == 'Both') &
        (pl.col('Rating') > 0) &
        (pl.col('2.0') == 1) &
        (pl.col('Map') == 'All maps')
    )
    .unique(['Match_id', 'Player_id', 'Date', 'Rating', 'Event_id', 'Map', 'Side'])
    .with_columns(
        pl.col('Date').cast(pl.Date).name.keep()
    )
    .sort('Player_id', 'Date', descending=False)
    .with_columns(
        time_gap_1=(pl.col('Date') - pl.col('Date').shift(1).over('Player_id')).fill_null(pl.duration(days=0)).dt.total_days(),
        time_gap_2=(pl.col('Date') - pl.col('Date').shift(2).over('Player_id')).fill_null(pl.duration(days=0)).dt.total_days(),
        time_gap_3=(pl.col('Date') - pl.col('Date').shift(3).over('Player_id')).fill_null(pl.duration(days=0)).dt.total_days(),
        team_performance=(pl.mean('Rating').over('Match_id', 'Team_id')*5-pl.col('Rating'))/4
    )
    .collect()
    .select('Team_id', 'Match_id', 'Player_id', 'Year', 'Rating', 'time_gap_1', 'time_gap_2', 'time_gap_3')
    .join(
        df_teams,
        ['Team_id', 'Match_id'],
        'inner'
    )
    .with_columns(
        matches=pl.n_unique('Match_id').over('Player_id')
    )
    .filter(
        (pl.col('matches') > 300)
    )
    .with_columns(
        pl.col('Team_id').cast(pl.String).cast(pl.Categorical),
        pl.col('Match_id').cast(pl.String).cast(pl.Categorical),
        pl.col('Player_id').cast(pl.String).cast(pl.Categorical),
        pl.col('best_of').cast(pl.String).cast(pl.Categorical),
        pl.col('Year').cast(pl.String).cast(pl.Categorical),
    )
    .with_columns(
        rating_lag_1=pl.col('Rating').shift(1).over('Player_id'),
        rating_lag_2=pl.col('Rating').shift(2).over('Player_id'),
        rating_lag_3=pl.col('Rating').shift(3).over('Player_id')
    )
    .with_columns(
        trend=(pl.col('date')-pl.col('date').min()).dt.total_days(),
        const=pl.lit(1),
    )
    .filter(
        ~(
        (pl.col('Team_id') == '7801') &
        (pl.col('Match_id') == '2329064') &
        (pl.col('Player_id') == '7382') &
        (pl.col('date') == datetime(2018, 11, 18)) &
        (pl.col('Rating') == 0.92)
        )
    )
)

players = players.rename({col: col.lower() for col in players.columns})

players.select(
    'match_id', 'player_id', 'year', 'rating', 'event_id', 'date', 'klaassen_dif', 'best_of',
    'elimination_match', 'prizepool', 'for_money', 'offline', 'rounds_to_end', 'log_rounds_to_end', 'tier_a',
    'tier_b', 'tier_c', 'tier_s', 'major', 'time_gap_1'
).write_csv('data_for_dominik.csv')

players.n_unique('player_id')
players.n_unique('match_id')
players.n_unique(['match_id', 'player_id'])
players.filter(players.select(['match_id', 'player_id']).is_duplicated())


for col in players.columns:
    players.select(col).describe()

## --- NICKS --- ###
with open('player_urls.txt', 'r') as file:
    data = file.read()

data = data.replace('\"','').split(',')

ids = [s.split('/')[2].strip() for s in data if len(s.split('/')) > 2]
nicks = [s.split('/')[3].strip() for s in data if len(s.split('/')) > 3]
nick_names = pl.DataFrame({'player_id': ids, 'nick': nicks}).with_columns(pl.col('player_id').cast(pl.Categorical))
players.select('player_id').join(nick_names, 'player_id', 'left').unique()['nick'].sort().to_list()
#------------------------------------------------------------------------------------------------------------------------#

df = df.with_columns(
    overperformance_base=pl.col('rating') - pl.col('predicted_skill'),
    overperformance_gap=pl.col('rating') - pl.col('predicted_skill_time_gap'),
    overperformance_h=pl.col('rating') - pl.col('predicted_skill_w_h'),
    overperformance_log=pl.col('rating') - pl.col('predicted_skill_w_log'),
    overperformance_sr=pl.col('rating') - pl.col('predicted_skill_w_sr'),
    overperformance_binary_gap=pl.when(pl.col('rating') > pl.col('predicted_skill_time_gap')).then(pl.lit(1)).otherwise(pl.lit(0)),
    overperformance_binary_h=pl.when(pl.col('rating') > pl.col('predicted_skill_w_h')).then(pl.lit(1)).otherwise(pl.lit(0)),
    overperformance_binary_log=pl.when(pl.col('rating') > pl.col('predicted_skill_w_log')).then(pl.lit(1)).otherwise(pl.lit(0)),
    overperformance_binary_sr=pl.when(pl.col('rating') > pl.col('predicted_skill_w_sr')).then(pl.lit(1)).otherwise(pl.lit(0))
)

player_ids = df['player_id'].unique()
n_runs = player_ids.shape[0]

### --- overperformance_base --- ###
# LOG
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

PARTS = 5
for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    gaps = player_df['time_gap_1'].abs().to_numpy()[1:]
    overperformance = player_df['overperformance_base'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    w = 1/np.log(gaps+np.e)
    sub_entropy = np.where(d_norm == 0, w*d_norm_when_0*np.log(d_norm_when_0), w*d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()
    
    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        gaps_perm = rng.permuted(np.tile(gaps, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, w*d_perm_norm_when_0*np.log(d_perm_norm_when_0), w*d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm

mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_log = np.array(list(pvals_log.values()))
p_values_log.sort()

rejects = np.flatnonzero(p_values_log < ALPHA).size

fwer = multipletests(p_values_log, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_log, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

labels = ['rejects', 'fwer_rejects', 'fdr_rejects', 'p_value']
values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]
max_label_length = max(len(label) for label in labels)

print(f'''
Rejects for entropy, log weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')


# LINEAR
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    gaps = player_df['time_gap_1'].abs().to_numpy()[1:]
    overperformance = player_df['overperformance_base'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    w = 1/(gaps+1)
    sub_entropy = np.where(d_norm == 0, w*d_norm_when_0*np.log(d_norm_when_0), w*d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()
    
    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        gaps_perm = rng.permuted(np.tile(gaps, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, w*d_perm_norm_when_0*np.log(d_perm_norm_when_0), w*d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm

mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_lin = np.array(list(pvals_log.values()))

rejects = np.flatnonzero(p_values_lin < ALPHA).size

fwer = multipletests(p_values_lin, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_lin, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]

print(f'''
Rejects for entropy, linear weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')

# SQRT
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    gaps = player_df['time_gap_1'].abs().to_numpy()[1:]
    overperformance = player_df['overperformance_base'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    w = 1/np.sqrt(gaps+1)
    sub_entropy = np.where(d_norm == 0, w*d_norm_when_0*np.log(d_norm_when_0), w*d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()

    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        gaps_perm = rng.permuted(np.tile(gaps, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, w*d_perm_norm_when_0*np.log(d_perm_norm_when_0), w*d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm

mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_sqrt = np.array(list(pvals_log.values()))

rejects = np.flatnonzero(p_values_sqrt < ALPHA).size

fwer = multipletests(p_values_sqrt, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_sqrt, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]

print(f'''
Rejects for entropy, sqrt weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')

# CONSTANT
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    overperformance = player_df['overperformance_base'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    w = 1
    sub_entropy = np.where(d_norm == 0, w*d_norm_when_0*np.log(d_norm_when_0), w*d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()

    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, w*d_perm_norm_when_0*np.log(d_perm_norm_when_0), w*d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm


mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_lin = np.array(list(pvals_log.values()))

rejects = np.flatnonzero(p_values_lin < ALPHA).size

fwer = multipletests(p_values_lin, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_lin, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]

print(f'''
Rejects for entropy, constant weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')
#------------------------------------------------------------------------------------------------------------------------#

### --- overperformance_gap --- ###
# LOG
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

PARTS = 5
for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    gaps = player_df['time_gap_1'].abs().to_numpy()[1:]
    overperformance = player_df['overperformance_log'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    sub_entropy = np.where(d_norm == 0, d_norm_when_0*np.log(d_norm_when_0), d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()
    
    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        gaps_perm = rng.permuted(np.tile(gaps, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, d_perm_norm_when_0*np.log(d_perm_norm_when_0), d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm

mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_log = np.array(list(pvals_log.values()))
p_values_log.sort()

rejects = np.flatnonzero(p_values_log < ALPHA).size

fwer = multipletests(p_values_log, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_log, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

labels = ['rejects', 'fwer_rejects', 'fdr_rejects', 'p_value']
values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]
max_label_length = max(len(label) for label in labels)

print(f'''
Rejects for entropy, log weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')


# LINEAR
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    gaps = player_df['time_gap_1'].abs().to_numpy()[1:]
    overperformance = player_df['overperformance_h'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    sub_entropy = np.where(d_norm == 0, d_norm_when_0*np.log(d_norm_when_0), d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()
    
    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        gaps_perm = rng.permuted(np.tile(gaps, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, d_perm_norm_when_0*np.log(d_perm_norm_when_0), d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm

mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_lin = np.array(list(pvals_log.values()))

rejects = np.flatnonzero(p_values_lin < ALPHA).size

fwer = multipletests(p_values_lin, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_lin, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]

print(f'''
Rejects for entropy, linear weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')

# SQRT
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    gaps = player_df['time_gap_1'].abs().to_numpy()[1:]
    overperformance = player_df['overperformance_sr'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    sub_entropy = np.where(d_norm == 0, d_norm_when_0*np.log(d_norm_when_0), d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()

    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        gaps_perm = rng.permuted(np.tile(gaps, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, d_perm_norm_when_0*np.log(d_perm_norm_when_0), d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm

mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_sqrt = np.array(list(pvals_log.values()))

rejects = np.flatnonzero(p_values_sqrt < ALPHA).size

fwer = multipletests(p_values_sqrt, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_sqrt, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]

print(f'''
Rejects for entropy, sqrt weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')

# CONSTANT
percs = {}
pvals_log = {}
rng = np.random.default_rng(SEED)

entropy_dists = np.zeros((n_runs,REPS))
entropies = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    overperformance = player_df['overperformance_gap'].to_numpy()

    d = np.abs(np.diff(overperformance))
    st_dev = np.std(d).item()
    d_sum = np.sum(d).item()
    d_norm = d/d_sum
    d_norm_when_0 = np.min(d_norm[d_norm > 0])/2

    sub_entropy = np.where(d_norm == 0, d_norm_when_0*np.log(d_norm_when_0), d_norm*np.log(d_norm))
    wh = -(np.sum(sub_entropy) / st_dev).item()

    sub_entropy_perms = []
    for part in range(PARTS):
        overperformance_perm = rng.permuted(np.tile(overperformance, (REPS//PARTS, 1)), axis=1)
        d_perm = np.abs(np.diff(overperformance_perm, axis=1))
        d_perm_norm = d_perm/d_sum
        d_perm_norm_when_0 = np.tile(np.array([np.min(row[row > 0]) for row in d_perm_norm]).reshape((REPS//PARTS,1))/2, d.size)

        sub_entropy_perm = np.where(d_perm_norm == 0, d_perm_norm_when_0*np.log(d_perm_norm_when_0), d_perm_norm*np.log(d_perm_norm))
        sub_entropy_perms.append(sub_entropy_perm)

    sub_entropy_perm = np.concatenate(sub_entropy_perms, axis=0)
    wh_perm = -np.sum(sub_entropy_perm, axis=1) / st_dev
    pperc = perc_sc(wh_perm, wh, kind='rank')
    entropies[i,] = wh
    pval = ne.evaluate('(100-pperc)/100')

    pvals_log[player] = np.around(pval, 6)
    entropy_dists[i,] = wh_perm


mean_entropy_dist = np.mean(entropy_dists, axis=0)
mean_entropy = np.mean(entropies, axis=0)
entropy_p_val = perc_sc(mean_entropy_dist, mean_entropy, kind='rank')
entropy_p_val_j = ne.evaluate('(100-entropy_p_val)/100')

p_values_lin = np.array(list(pvals_log.values()))

rejects = np.flatnonzero(p_values_lin < ALPHA).size

fwer = multipletests(p_values_lin, alpha=ALPHA, method='holm-sidak')[0]
fwer_rejects = np.flatnonzero(fwer).size

FDR_p = fdrcorrection(p_values_lin, alpha = ALPHA, method = 'indep')[0]
fdr_rejects = np.flatnonzero(FDR_p).size

values = [rejects, fwer_rejects, fdr_rejects, entropy_p_val_j]

print(f'''
Rejects for entropy, constant weight
{labels[0]:>{max_label_length}}: {values[0]}
{labels[1]:>{max_label_length}}: {values[1]}
{labels[2]:>{max_label_length}}: {values[2]}
{labels[3]:>{max_label_length}}: {values[3]}
''')

### --- overperformance_binary_gap --- ###
faulty_shots = []
p_vals_dep = np.zeros((n_runs, 3))
D_hat = np.zeros(n_runs)
P_hat_p = np.zeros(n_runs)
entropies = np.zeros(n_runs)
p_vals_ent = np.zeros(n_runs)
t_stats_Pp_dist = np.zeros((n_runs,REPS))
t_stats_D_dist = np.zeros((n_runs,REPS))
entropy_dists = np.zeros((n_runs,REPS))

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    arr = player_df['overperformance_binary_gap'].to_numpy()
    # DEPENDENCE SIMULTANEOUS
    try:
        D_had_p_value, P_hat_p_p_value = get_p_values(arr, reps=REPS, k=K)
        p_vals_dep[i,0] = D_had_p_value
        p_vals_dep[i,1] = P_hat_p_p_value
    except ValueError:
        p_vals_dep[i,0] = np.nan
        p_vals_dep[i,1] = np.nan
    p_vals_dep[i,2] = i

    # ENTROPY SIMULTANEOUS
    H_s = weighted_entropy(arr)
    H_s_dist = weighted_entropy_dist(shots=arr, reps=REPS, seed=42)
    p_vals_ent[i] = entropy_pval(stat=H_s, dist=H_s_dist)
    # ENTROPY JOINT
    entropy_dists[i,] = H_s_dist
    entropies[i,] = H_s


p_vals_notnan = p_vals_dep[~np.isnan(p_vals_dep).any(axis=1)]

D_had_p_values = np.sort(p_vals_notnan[:,0])
P_hat_p_p_value = np.sort(p_vals_notnan[:,1])

hot_hands_D = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=False)
hot_hands_P_p = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=False)
hot_hands_D_sidak = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=True)
hot_hands_P_p_sidak = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=True)
hot_hands_D_FDR = get_FDR(p_values=D_had_p_values, alpha=0.05)
hot_hands_P_p_FDR = get_FDR(p_values=P_hat_p_p_value, alpha=0.05)
hot_hands_ent = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=False)
hot_hands_ent_sidak = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=True)
hot_hands_ent_FDR = get_FDR(p_values=p_vals_ent, alpha=0.05)
#------------------------------------------------------------------------------------------------------------------------#

## --- PRINTS --- ###
print(f'''\n
Tested runs:    {p_vals_notnan.shape[0]}\n

<<< --- DEPENDENCE --- >>>\n
SIMULTANEOUS
 Individual   <D, P>:   <{hot_hands_D}, {hot_hands_P_p}>
       FWER   <D, P>:   <{hot_hands_D_sidak}, {hot_hands_P_p_sidak}>
        FDR   <D, P>:   <{hot_hands_D_FDR}, {hot_hands_P_p_FDR}>\n

<<< --- ENTROPY --- >>>\n

SIMULTANEOUS
          Individual: {hot_hands_ent}
                FWER: {hot_hands_ent_sidak}
                 FDR: {hot_hands_ent_FDR}\n
''')

### --- overperformance_binary_gap_h --- ###
faulty_shots = []
p_vals_dep = np.zeros((n_runs, 3))
D_hat = np.zeros(n_runs)
P_hat_p = np.zeros(n_runs)
entropies = np.zeros(n_runs)
p_vals_ent = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    arr = player_df['overperformance_binary_h'].to_numpy()
    # DEPENDENCE SIMULTANEOUS
    try:
        D_had_p_value, P_hat_p_p_value = get_p_values(arr, reps=REPS, k=K)
        p_vals_dep[i,0] = D_had_p_value
        p_vals_dep[i,1] = P_hat_p_p_value
    except ValueError:
        p_vals_dep[i,0] = np.nan
        p_vals_dep[i,1] = np.nan
        # faulty_shots.append(np_shots[i,])
    p_vals_dep[i,2] = i

    # ENTROPY SIMULTANEOUS
    H_s = weighted_entropy(arr)
    H_s_dist = weighted_entropy_dist(shots=arr, reps=REPS, seed=42)
    p_vals_ent[i] = entropy_pval(stat=H_s, dist=H_s_dist)
    # ENTROPY JOINT
    entropy_dists[i,] = H_s_dist
    entropies[i,] = H_s


p_vals_notnan = p_vals_dep[~np.isnan(p_vals_dep).any(axis=1)]

D_had_p_values = np.sort(p_vals_notnan[:,0])
P_hat_p_p_value = np.sort(p_vals_notnan[:,1])

hot_hands_D = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=False)
hot_hands_P_p = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=False)
hot_hands_D_sidak = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=True)
hot_hands_P_p_sidak = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=True)
hot_hands_D_FDR = get_FDR(p_values=D_had_p_values, alpha=0.05)
hot_hands_P_p_FDR = get_FDR(p_values=P_hat_p_p_value, alpha=0.05)
hot_hands_ent = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=False)
hot_hands_ent_sidak = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=True)
hot_hands_ent_FDR = get_FDR(p_values=p_vals_ent, alpha=0.05)
#------------------------------------------------------------------------------------------------------------------------#

## --- PRINTS --- ###
print(f'''\n
Tested runs:    {p_vals_notnan.shape[0]}\n

<<< --- DEPENDENCE --- >>>\n
SIMULTANEOUS
 Individual   <D, P>:   <{hot_hands_D}, {hot_hands_P_p}>
       FWER   <D, P>:   <{hot_hands_D_sidak}, {hot_hands_P_p_sidak}>
        FDR   <D, P>:   <{hot_hands_D_FDR}, {hot_hands_P_p_FDR}>\n

<<< --- ENTROPY --- >>>\n

SIMULTANEOUS
          Individual: {hot_hands_ent}
                FWER: {hot_hands_ent_sidak}
                 FDR: {hot_hands_ent_FDR}\n
''')


### --- overperformance_binary_gap_sr --- ###
faulty_shots = []
p_vals_dep = np.zeros((n_runs, 3))
D_hat = np.zeros(n_runs)
P_hat_p = np.zeros(n_runs)
entropies = np.zeros(n_runs)
p_vals_ent = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    arr = player_df['overperformance_binary_sr'].to_numpy()
    # DEPENDENCE SIMULTANEOUS
    try:
        D_had_p_value, P_hat_p_p_value = get_p_values(arr, reps=REPS, k=K)
        p_vals_dep[i,0] = D_had_p_value
        p_vals_dep[i,1] = P_hat_p_p_value
    except ValueError:
        p_vals_dep[i,0] = np.nan
        p_vals_dep[i,1] = np.nan
    p_vals_dep[i,2] = i

    # ENTROPY SIMULTANEOUS
    H_s = weighted_entropy(arr)
    H_s_dist = weighted_entropy_dist(shots=arr, reps=REPS, seed=42)
    p_vals_ent[i] = entropy_pval(stat=H_s, dist=H_s_dist)
    # ENTROPY JOINT
    entropy_dists[i,] = H_s_dist
    entropies[i,] = H_s


p_vals_notnan = p_vals_dep[~np.isnan(p_vals_dep).any(axis=1)]

D_had_p_values = np.sort(p_vals_notnan[:,0])
P_hat_p_p_value = np.sort(p_vals_notnan[:,1])

hot_hands_D = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=False)
hot_hands_P_p = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=False)
hot_hands_D_sidak = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=True)
hot_hands_P_p_sidak = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=True)
hot_hands_D_FDR = get_FDR(p_values=D_had_p_values, alpha=0.05)
hot_hands_P_p_FDR = get_FDR(p_values=P_hat_p_p_value, alpha=0.05)
hot_hands_ent = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=False)
hot_hands_ent_sidak = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=True)
hot_hands_ent_FDR = get_FDR(p_values=p_vals_ent, alpha=0.05)
#------------------------------------------------------------------------------------------------------------------------#

## --- PRINTS --- ###
print(f'''\n
Tested runs:    {p_vals_notnan.shape[0]}\n

<<< --- DEPENDENCE --- >>>\n
SIMULTANEOUS
 Individual   <D, P>:   <{hot_hands_D}, {hot_hands_P_p}>
       FWER   <D, P>:   <{hot_hands_D_sidak}, {hot_hands_P_p_sidak}>
        FDR   <D, P>:   <{hot_hands_D_FDR}, {hot_hands_P_p_FDR}>\n

<<< --- ENTROPY --- >>>\n

SIMULTANEOUS
          Individual: {hot_hands_ent}
                FWER: {hot_hands_ent_sidak}
                 FDR: {hot_hands_ent_FDR}\n
''')



### --- overperformance_binary_gap_log --- ###
faulty_shots = []
p_vals_dep = np.zeros((n_runs, 3))
D_hat = np.zeros(n_runs)
P_hat_p = np.zeros(n_runs)
entropies = np.zeros(n_runs)
p_vals_ent = np.zeros(n_runs)

for i, player in enumerate(player_ids):
    print(i)
    player_df = df.filter(pl.col('player_id') == player)
    arr = player_df['overperformance_binary_log'].to_numpy()
    # DEPENDENCE SIMULTANEOUS
    try:
        D_had_p_value, P_hat_p_p_value = get_p_values(arr, reps=REPS, k=K)
        p_vals_dep[i,0] = D_had_p_value
        p_vals_dep[i,1] = P_hat_p_p_value
    except ValueError:
        p_vals_dep[i,0] = np.nan
        p_vals_dep[i,1] = np.nan
    p_vals_dep[i,2] = i

    # ENTROPY SIMULTANEOUS
    H_s = weighted_entropy(arr)
    H_s_dist = weighted_entropy_dist(shots=arr, reps=REPS, seed=42)
    p_vals_ent[i] = entropy_pval(stat=H_s, dist=H_s_dist)
    # ENTROPY JOINT
    entropy_dists[i,] = H_s_dist
    entropies[i,] = H_s


p_vals_notnan = p_vals_dep[~np.isnan(p_vals_dep).any(axis=1)]

D_had_p_values = np.sort(p_vals_notnan[:,0])
P_hat_p_p_value = np.sort(p_vals_notnan[:,1])

hot_hands_D = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=False)
hot_hands_P_p = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=False)
hot_hands_D_sidak = get_n_rejets(p_values=D_had_p_values, alpha=0.05, sidak=True)
hot_hands_P_p_sidak = get_n_rejets(p_values=P_hat_p_p_value, alpha=0.05, sidak=True)
hot_hands_D_FDR = get_FDR(p_values=D_had_p_values, alpha=0.05)
hot_hands_P_p_FDR = get_FDR(p_values=P_hat_p_p_value, alpha=0.05)
hot_hands_ent = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=False)
hot_hands_ent_sidak = get_n_rejets(p_values=p_vals_ent, alpha=0.05, sidak=True)
hot_hands_ent_FDR = get_FDR(p_values=p_vals_ent, alpha=0.05)
#------------------------------------------------------------------------------------------------------------------------#

## --- PRINTS --- ###
print(f'''\n
Tested runs:    {p_vals_notnan.shape[0]}\n

<<< --- DEPENDENCE --- >>>\n
SIMULTANEOUS
 Individual   <D, P>:   <{hot_hands_D}, {hot_hands_P_p}>
       FWER   <D, P>:   <{hot_hands_D_sidak}, {hot_hands_P_p_sidak}>
        FDR   <D, P>:   <{hot_hands_D_FDR}, {hot_hands_P_p_FDR}>\n

<<< --- ENTROPY --- >>>\n

SIMULTANEOUS
          Individual: {hot_hands_ent}
                FWER: {hot_hands_ent_sidak}
                 FDR: {hot_hands_ent_FDR}\n
''')












