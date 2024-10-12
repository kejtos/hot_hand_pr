from functions import *

## --- SETUP --- ###
DIR = 'C:\\Users\\rejth\\OneDrive - kejtos\\Connection\\Plocha\\Doktorát\\Hot hand\\Data\\'
TILE_FRENZIES = ['Tile Frenzy', 'Tile Frenzy Mini']
SCEN_TYPES = 'all' # flicks, tiles, all
#------------------------------------------------------------------------------------------------------------------------#

## --- RETRIEVING THE DATA --- ###
list_zipfiles = get_zipfile_names(directory=DIR)

list_sets_scenario_names = []
players = []
n_scen = 0

for player, zipname in enumerate(list_zipfiles, 1):
    list_of_names = get_scenario_names(zipname=zipname)
    scenario_names = sorted({re.search('(.*)(?= - Challenge)', name)[0] for name in list_of_names})
    players.append(player)
    list_sets_scenario_names.append(scenario_names)
    n_scen += len(list_of_names)

dict_scenario_names = {player: name for player, name in zip(players, list_sets_scenario_names)}

if SCEN_TYPES in ('flicks', 'all'):
    list_of_regs = ['(?i)reflex', '(?i)flick', '(?i)microshot','(?i)micro shot','(?i)microshots','(?i)micro shots', '(?i)speed', '(?i)timing', '(?i)popcorn']
    flick_scens = set()
    for key, names in dict_scenario_names.items():
        for name in names:
            for regex in list_of_regs:
                if re.search(regex, name):
                    flick_scens.add(name)

    flick_delete = [
        r'Reflex Poke-Micro++ Regenerating Goated', r'VT Multiclick 120 Intermediate Speed Focus LG56', r'Close flick v2', r"KovaaK's League S6 - beanClick 250% Speed", r'Flicker Plaza No Strafes Dash Master',
        r'Floating Heads Timing Easy', r'Fireworks Flick [x-small]', r'Target Acquisition Flick Small 400ms', r'Houston Outlaws - Speed Switching', r'psalmflick TPS Strafing',
        r'VT Pasu Rasp Intermediate Speed Focus LG56', r'Target Acquisition Flick Easy', r'Valorant Small Flicks', r'Flicker Plaza Hard', r'Flicker XYZ', r'Flip off Flick Random',
        r'Jumbo Tile Frenzy Flick 180', r'Flicker Plaza', r'Tamspeed 2bpes', r'e1se Braking Reflex Flick EASY', r'Target Acquisition Flick', r'VSS GP9 +50% speed', r'Floating Heads Timing 400% no mag',
        r'Reflex Tracking a+', r'Target Acquisition Flick 350ms', r'Flicker XYZ Easy', r'FuglaaXYZ Voltaic No Blinks but the bot has taken a speed potion', r'KovaaK_s League S6 - beanClick 250_ Speed',
        r'Tamspeed 2bp Bronze', r'Floating Heads Timing 400_', r'Ground Plaza Voltaic 1 Invincible Always Dash Speed No Dash', r'Flicker Plaza Grandmaster', r'1wall2targets_smallflicks 60s',
        r'PlasmaFlick 180', r'Reactive Flick to Track', r'Target Acquisition Flick Small', r'Floating Heads Timing 400%', r'Eclipse Flick Warmup', r'Popcorn Goated TI easy', r'Popcorn Gauntlet Raspberry Master'
    ]

    flick_scens = [scen for scen in flick_scens if not re.search('(?i)tracking', scen)]
    flick_scens = [scen for scen in flick_scens if not re.search('(?i)smooth', scen)]
    flick_scens = [scen for scen in flick_scens if scen not in flick_delete]
    if SCEN_TYPES == 'all':
        all_scens = flick_scens + TILE_FRENZIES
        df = create_df_from_zipcsvs(list_zipfiles=list_zipfiles, names_of_scenarios=all_scens)
    else:
        df = create_df_from_zipcsvs(list_zipfiles=list_zipfiles, names_of_scenarios=flick_scens)
elif SCEN_TYPES == 'tiles':
    df = create_df_from_zipcsvs(list_zipfiles=list_zipfiles, names_of_scenarios=TILE_FRENZIES)

df = df[['String_of_shots', 'Number_of_shots', 'Hits', 'Misses', 'Accuracy']].reset_index(drop=True)

df['String_of_shots'] = df['String_of_shots'].apply(lambda x: ','.join(map(str, x)))
df.to_csv('kovaak_data.csv', header=True, index=False)