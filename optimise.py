from requests import get
from pandas import DataFrame as pdf, read_csv, concat, Series, set_option, options, option_context
from numpy import arange, array
import cvxpy

options.display.width = 0
set_option('display.max_columns', None)
#set_option('display.max_rows', None)

def get_player(x): return xpts.loc[x[0], 'Name']
def decay(row, rate): return row * rate**arange(0, len(row))
def x90(mins): return 0.000215171*(mins**2) - 0.04515889*mins + 2.342686553 + 1

base_path = '/Users/desmundhui/Downloads/'
xpts = read_csv(base_path + 'fplreview.csv')
xpts['Pos'] = xpts['Pos'].astype("category").cat.set_categories(['F', 'M', 'D', 'G'])

#approximate points assuming all xMin=90
xpts_90 = xpts.copy()
for i in range(5, xpts_90.shape[1], 2):
    xpts_90.iloc[:, i+1] *= xpts_90.iloc[:, i].map(x90)

#decay([1,1,1,1,1,1,1,1], 0.84)
#adjustment for form of player, using a factor of 0.8
#form = read_csv(base_path + 'fplreview_formFactor.csv', encoding= 'unicode_escape')
#form['FF_0.8'] = (((form['Form Factor'] - 1) * 0.8) + 1).fillna(1)

#check size of dataframe first
#if xpts.shape[0] != form.shape[0]:
#    print('New player(s) in fplreview.csv, please update form.csv first.')
#    quit()

#cvxpy
def optim(itb=100, incl=[], excl=[], preset=[], weeks_from=1, weeks_to=8, ft=1, use_form=False, use_90=False,
    bench_weight=[0, 0.2, 0.05, 0], BB=0, TC=0, WC=0, FH=0, decay_rate=0.84):
    global xpts
    #use form to estimate
    #if use_form: xpts.iloc[:, 5:] = xpts.iloc[:, 5:].mul(form['FF_0.8'], axis=0)
    #use 90xMin
    if use_90: xpts = xpts_90
    #Isolate the targeted GWs, and eliminate poor value options
    pts_df = xpts.filter(like='_Pts', axis=1).iloc[:,weeks_from-1:weeks_to]
    xpts['total'] = pts_df.sum(axis=1)
    #include v cheap players that may offer no value (in terms of ppm)
    cheap = xpts[xpts['SV']<=4.5].sort_values(['SV', 'total'], ascending=[True, False])
    cheap_idx = (
        cheap[cheap['SV']<=4].groupby(['Pos', 'SV']).head(5).index.union(  #take first 5 G & D of each price point <= 4.0
            cheap[(cheap['Pos']=='M') | (cheap['Pos']=='F')].groupby(['Pos', 'SV']).head(5).index #M & D, <= 4.5
        )
    )

    xpts['value'] = xpts['total'] / xpts['SV'] / pts_df.shape[1]
    xpts2 = concat([xpts[xpts['value'] > (0.35 if not use_90 else 0.5)],xpts.loc[cheap_idx, :],xpts.iloc[preset, :]]).drop_duplicates()
    pts_df2 = xpts2.filter(like='_Pts', axis=1).iloc[:,weeks_from-1:weeks_to]
    #time decay
    if decay_rate != 1:
        pts_df2 = pts_df2.apply(decay, axis=1, args=(decay_rate,))
        undecay = (1/decay_rate)**arange(0, 8)
        undecay_total = 0

    GW = int(pts_df2.columns[0][:-4]) #initial GW
    weeks_to = min(weeks_to, 38+weeks_from-GW) #in case weeks_to exceed GW38

    pts_dict = pts_df2.to_dict('list')
    sell_price = xpts2['SV'].tolist()
    buy_price = xpts2['BV'].tolist()

    var_dict = {}
    price_dict = {}
    total_points = 0
    constraints = []

    var_dict[f'{GW}_FT'] = ft
    var_dict[f'{GW}_itb'] = itb

    for gw in range(GW, GW - weeks_from + weeks_to + 1):#gw=34
        #variable construction
        var_dict[f'{gw}_selction'] = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_bench_0']  = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_bench_1']  = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_bench_2']  = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_bench_3']  = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_squad']    = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_captain']  = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_auxiliary']= cvxpy.Variable(boolean=True) #refer to Optimization for FPL (with Excel) - Part 3
        var_dict[f'{gw}_tra_used'] = cvxpy.Variable(integer=True)
        var_dict[f'{gw}_incoming'] = cvxpy.Variable(xpts2.shape[0], boolean=True)
        var_dict[f'{gw}_outgoing'] = cvxpy.Variable(xpts2.shape[0], boolean=True)

        #price dict
        price_dict[f'{gw}_sold_amt'] = cvxpy.sum(sell_price @ var_dict[f'{gw}_outgoing'])
        price_dict[f'{gw}_bought_amt']  = cvxpy.sum(buy_price @ var_dict[f'{gw}_incoming'])

        #construct total_points as target to maximise
        total_points += (
            pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_selction'] +
            pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_captain'] +
            (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_0']) * bench_weight[0] +
            (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_1']) * bench_weight[1] +
            (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_2']) * bench_weight[2] +
            (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_3']) * bench_weight[3]
        )
        #Add the remaining weight of bench if BB
        if gw==GW+BB-weeks_from:
            total_points += (
                (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_0']) * (1-bench_weight[0]) +
                (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_1']) * (1-bench_weight[1]) +
                (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_2']) * (1-bench_weight[2]) +
                (pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_bench_3']) * (1-bench_weight[3])
            )
        #TC
        elif gw==GW+TC-weeks_from: total_points += pts_dict[f'{gw}_Pts'] @ var_dict[f'{gw}_captain']

        #total_points deduction for too many transfers
        #WC or FH or no preset (i.e. a WC)
        if gw==GW+WC-weeks_from or gw==GW+FH-weeks_from or (not preset and gw==GW):
            #no transfer cost when WC/FH, so no deduction from total_points. Next GW always 1 FT
            var_dict[f'{gw+1}_FT'] = 1
            var_dict[f'{gw+1}_itb'] = var_dict[f'{gw}_itb'] + price_dict[f'{gw}_sold_amt'] - price_dict[f'{gw}_bought_amt']
        #in week FH+1, compare itb with week FH-1
        elif (FH!=0 and gw==GW+FH+1-weeks_from):
            total_points -= cvxpy.maximum((var_dict[f'{gw}_tra_used'] - var_dict[f'{gw}_FT']) * 4, 0)
            var_dict[f'{gw+1}_FT'] = var_dict[f'{gw}_auxiliary'] + 1
            var_dict[f'{gw+1}_itb'] = var_dict[f'{gw-1}_itb'] + price_dict[f'{gw}_sold_amt'] - price_dict[f'{gw}_bought_amt']
        else:
            total_points -= cvxpy.maximum((var_dict[f'{gw}_tra_used'] - var_dict[f'{gw}_FT']) * 4, 0)
            var_dict[f'{gw+1}_FT'] = var_dict[f'{gw}_auxiliary'] + 1
            var_dict[f'{gw+1}_itb'] = var_dict[f'{gw}_itb'] + price_dict[f'{gw}_sold_amt'] - price_dict[f'{gw}_bought_amt']

        #give more weight to rolling FT
        total_points += (var_dict[f'{gw}_FT']-1) * 0.5 #so 0.5 pts if you have 2FTs, else 0
        #doesnt matter if it deducts points when ft==0, since further weeks' results wont change

        #constraints
        #each team cannot have more than 3 players
        for team in set(xpts2['Team']):
            constraints += [array(xpts2['Team'] == team) @ var_dict[f'{gw}_squad'] <= 3]

        constraints += [
            var_dict[f'{gw}_squad'] == ( #so player cant be selected AND benched
                var_dict[f'{gw}_selction'] + var_dict[f'{gw}_bench_0'] + var_dict[f'{gw}_bench_1'] +
                var_dict[f'{gw}_bench_2'] + var_dict[f'{gw}_bench_3']
            ),
            sum(var_dict[f'{gw}_bench_0']) == 1,
            array(xpts2['Pos'] == 'G') @ var_dict[f'{gw}_bench_0'] == 1, #bench_0 = GK
            sum(var_dict[f'{gw}_bench_1']) == 1,
            sum(var_dict[f'{gw}_bench_2']) == 1,
            sum(var_dict[f'{gw}_bench_3']) == 1,
            sum(var_dict[f'{gw}_captain']) == 1,
            sum(cvxpy.pos(var_dict[f'{gw}_selction'] - var_dict[f'{gw}_captain'])) <= 10, #always 10 noncaptains from XI, fit DCP rule
            var_dict[f'{gw}_tra_used'] >= 0, #no negative transfers
            #squad limits
            array(xpts2['Pos'] == 'G') @ var_dict[f'{gw}_squad'] == 2,
            array(xpts2['Pos'] == 'D') @ var_dict[f'{gw}_squad'] == 5,
            array(xpts2['Pos'] == 'M') @ var_dict[f'{gw}_squad'] == 5,
            array(xpts2['Pos'] == 'F') @ var_dict[f'{gw}_squad'] == 3,
            #XI limits
            array(xpts2['Pos'] == 'G') @ var_dict[f'{gw}_selction'] == 1,
            array(xpts2['Pos'] == 'D') @ var_dict[f'{gw}_selction'] >= 3,
            array(xpts2['Pos'] == 'M') @ var_dict[f'{gw}_selction'] >= 2,
            array(xpts2['Pos'] == 'F') @ var_dict[f'{gw}_selction'] >= 1,
            #only 10 outfielders
            array(xpts2['Pos'] != 'G') @ var_dict[f'{gw}_selction'] == 10,

            #auxiliary for FT logic (cr. @Sertalpbial)
            var_dict[f'{gw}_FT'] - var_dict[f'{gw}_tra_used'] <= 2 * var_dict[f'{gw}_auxiliary'],
            var_dict[f'{gw}_FT'] - var_dict[f'{gw}_tra_used'] >= var_dict[f'{gw}_auxiliary'] + (-14) * (1-var_dict[f'{gw}_auxiliary'])
        ]

    #compare last week squad to this week squad, ensuring no. of transfers made and itb are correct
        constraints += [
            cvxpy.sum(var_dict[f'{gw}_incoming']) == var_dict[f'{gw}_tra_used'],
            var_dict[f'{gw+1}_itb'] >= 0
        ]
        if (FH!=0 and gw==GW+FH+1-weeks_from): #make sure in the week after FH, you get back original squad + transfer
            constraints += [
                var_dict[f'{gw}_squad'] == (
                    var_dict[f'{gw-2}_squad'] + var_dict[f'{gw}_incoming'] - var_dict[f'{gw}_outgoing']
            )]
        else:
            constraints += [
                var_dict[f'{gw}_squad'] == (
                    ([idx in preset for idx in xpts2.index.values] if gw==GW else var_dict[f'{gw-1}_squad']) +
                        var_dict[f'{gw}_incoming'] - var_dict[f'{gw}_outgoing']
            )]

    #force incl/excl player
    for id, gw_from_start in excl:
        constraints += [(xpts2.index==id) @ var_dict[f'{GW - weeks_from + gw_from_start}_squad'] == 0]
    for id, gw_from_start in incl:
        constraints += [(xpts2.index==id) @ var_dict[f'{GW - weeks_from + gw_from_start}_squad'] == 1]

    problem = cvxpy.Problem(cvxpy.Maximize(total_points), constraints)
    problem.solve(solver=cvxpy.CBC, allowablePercentageGap=0.1, verbose=False)

    if problem.status in ["infeasible", "unbounded"]:
        print(problem.status)
        quit()

    print(f'\nObjective value between GW{GW}-GW{GW - weeks_from + weeks_to} is {round(problem.value, 3)}. (Decay at {decay_rate})')
    if preset:  print(f'Preset used.')
    #if use_form: print(f'Form considered.')
    if incl: print(f'Force include: {", ".join(map(get_player, incl))}')
    if excl: print(f'Force exclude: {", ".join(map(get_player, excl))}')

    for gw in range(GW, GW - weeks_from + weeks_to + 1):
        res = concat([xpts2.iloc[:, :5], pts_df2[f'{gw}_Pts']], axis=1).reset_index(drop=True)
        #transfers in/out
        if gw==GW or (gw==GW + FH and FH==1): #no gw-1 for first week OR week 1 FH (so no gw-2)
            tra = concat([res, Series((var_dict[f'{gw}_squad'].value -
                [idx in preset for idx in xpts2.index.values]).round(), name='tra')],axis=1)
        elif gw==GW+FH-weeks_from+1: #for the week after FH is used, compare squad with 2 weeks ago
            tra = concat([res, Series((var_dict[f'{gw}_squad'].value - var_dict[f'{gw-2}_squad'].value).round(), name='tra')], axis=1)
        else:
            tra = concat([res, Series((var_dict[f'{gw}_squad'].value - var_dict[f'{gw-1}_squad'].value).round(), name='tra')], axis=1)
        tra = tra.sort_values(by='Pos', ascending=False)

        res = concat([res,
            Series(var_dict[f'{gw}_selction'].value, name=f'{gw}_Select'),
            Series(var_dict[f'{gw}_captain'].value, name=f'{gw}_Cap'),
            Series(var_dict[f'{gw}_squad'].value, name=f'{gw}_Squad'),
            Series(var_dict[f'{gw}_bench_0'].value, name=f'{gw}_Bench_0'),
            Series(var_dict[f'{gw}_bench_1'].value, name=f'{gw}_Bench_1'),
            Series(var_dict[f'{gw}_bench_2'].value, name=f'{gw}_Bench_2'),
            Series(var_dict[f'{gw}_bench_3'].value, name=f'{gw}_Bench_3'),
        ], axis=1)
        #select elements that are not equal to 0, then sort by selected, Pos
        res = res[res.iloc[:, 6:].ne(0).any(1)].sort_values(by=[res.columns[6], *res.columns[9:12], 'Pos'], ascending=False).iloc[:,:-4]
        prediction = res.iloc[:11, -4] @ (res.iloc[:11,-2]+1) #dot prod of pts and captain + 1 (so captain becomes 2)
        pred_bench = res.iloc[-4:,-4].sum()
        if decay_rate != 1:
            undecay_pred  = round(prediction * undecay[gw-GW], 3)
            undecay_bench = round(pred_bench * undecay[gw-GW], 3)

        if gw==GW and not preset: #for first week WC
            print(
                f'\nxPts: {round(prediction, 3)} ' + '(0) '+ f'Bench Points: {round(pred_bench, 3)} ' +
                f'TV used: {round(res["SV"].sum(), 1)} ' + 'WC'
            )
            if decay_rate != 1:
                print(f'un-decay xPts: {undecay_pred} Bench: {undecay_bench}')
                undecay_total += undecay_pred
            print(f'Transfer(s) in: {", ".join(tra[tra["tra"].eq(1)]["Name"])}') #no transfer out as no preset
            print(res.reset_index(drop=True))
        elif gw==GW+WC-weeks_from:
            print(
                f'\nxPts: {round(prediction, 3)} ' + '(0) '+ f'Bench Points: {round(pred_bench, 3)} ' +
                f'TV used: {round(res["SV"].sum(), 1)} ' +
                f'Transfer: {var_dict[f"{gw}_tra_used"].value.round()} ' + 'WC'
            )
            if decay_rate != 1:
                print(f'un-decay xPts: {undecay_pred} Bench: {undecay_bench}')
                undecay_total += undecay_pred
            print(f'Transfer(s) out: {", ".join(tra[tra["tra"].eq(-1)]["Name"])}')
            print(f'Transfer(s) in: {", ".join(tra[tra["tra"].eq(1)]["Name"])}')
            print(res.reset_index(drop=True))
        elif gw==GW+FH-weeks_from:
            print(
                f'\nxPts: {round(prediction, 3)} ' + '(0) '+
                f'Bench Points: {round(pred_bench, 3)} ' +
                f'TV used: {round(res["SV"].sum(), 1)} ' + 'FH'
            )
            if decay_rate != 1:
                print(f'un-decay xPts: {undecay_pred} Bench: {undecay_bench}')
                undecay_total += undecay_pred
            print(f'Transfer(s) out: {", ".join(tra[tra["tra"].eq(-1)]["Name"])}')
            print(f'Transfer(s) in: {", ".join(tra[tra["tra"].eq(1)]["Name"])}')
            print(res.reset_index(drop=True))
        elif gw==GW+BB-weeks_from:
            prediction = res.iloc[:, -4] @ (res.iloc[:,-2] + 1) #included bench
            try: pts_hit = round(max(var_dict[f"{gw}_tra_used"].value - var_dict[f'{gw}_FT'].value, 0) * -4) #pts hit
            except AttributeError: pts_hit = round(max(var_dict[f"{gw}_tra_used"].value - ft, 0) * -4)
            print(
                f'\nxPts: {round(prediction, 3)} ' + #included bench
                f'({pts_hit}) ' + f'Bench Boost TV used: {round(res["SV"].sum(), 1)} ' +
                f'Transfer: {var_dict[f"{gw}_tra_used"].value.round()}'
            )
            if decay_rate != 1:
                print(f'un-decay xPts: {round(prediction * undecay[gw-GW], 3)}')
                undecay_total += round(prediction * undecay[gw-GW], 3) + pts_hit
            print(f'Transfer(s) out: {", ".join(tra[tra["tra"].eq(-1)]["Name"])}')
            print(f'Transfer(s) in: {", ".join(tra[tra["tra"].eq(1)]["Name"])}')
            print(res.sort_values(by=[res.columns[5]], ascending=False).reset_index(drop=True))
        elif gw==GW+TC-weeks_from:
            prediction = res.iloc[:, -4] @ (2*(res.iloc[:,-2])+1) #2*captaincy + all 1
            try: pts_hit = round(max(var_dict[f"{gw}_tra_used"].value - var_dict[f'{gw}_FT'].value, 0) * -4) #pts hit
            except AttributeError: pts_hit = round(max(var_dict[f"{gw}_tra_used"].value - ft, 0) * -4)
            print(
                f'\nxPts: {round(prediction, 3)} ' + f'({pts_hit}) ' +
                f'Bench Points: {round(pred_bench, 3)} ' +
                f'TV used: {round(res["SV"].sum(), 1)} ' +
                f'Transfer: {var_dict[f"{gw}_tra_used"].value.round()} ' + 'Triple Cap'
            )
            if decay_rate != 1:
                print(
                    f'un-decay xPts: {round(prediction * undecay[gw-GW], 3)} ' +
                    f'Bench: {undecay_bench}'
                )
                undecay_total += round(prediction * undecay[gw-GW], 3) + pts_hit
            print(f'Transfer(s) out: {", ".join(tra[tra["tra"].eq(-1)]["Name"])}')
            print(f'Transfer(s) in: {", ".join(tra[tra["tra"].eq(1)]["Name"])}')
            print(res.reset_index(drop=True))
        else:
            try: pts_hit = round(max(var_dict[f"{gw}_tra_used"].value - var_dict[f'{gw}_FT'].value, 0) * -4) #pts hit
            except AttributeError: pts_hit = round(max(var_dict[f"{gw}_tra_used"].value - ft, 0) * -4)
            print(
                f'\nxPts: {round(prediction, 3)} ' + f'({pts_hit}) ' +
                f'Bench Points: {round(pred_bench, 3)} ' +
                f'TV used: {round(res["SV"].sum(), 1)} ' +
                f'Transfer: {var_dict[f"{gw}_tra_used"].value.round()}'
            )
            if decay_rate != 1:
                print(f'un-decay xPts: {undecay_pred} Bench: {undecay_bench}')
                undecay_total += round(prediction * undecay[gw-GW], 3) + pts_hit
            print(f'Transfer(s) out: {", ".join(tra[tra["tra"].eq(-1)]["Name"])}')
            print(f'Transfer(s) in: {", ".join(tra[tra["tra"].eq(1)]["Name"])}')
            print(res.reset_index(drop=True))

    print("Total xPts:", undecay_total)

#with option_context('display.max_rows', None):
#    print(xpts_90.iloc[:,:7].sort_values(by=xpts_90.columns[6], ascending=False))
#optim(101.1, preset=[11, 258, 103, 541, 389, 253, 300, 467, 387, 201, 232, 522, 23, 66, 60], FH=3)
#optim(itb=0.6, preset=[11, 258, 103, 541, 389, 253, 300, 467, 387, 201, 232, 522, 23, 66, 60])
#optim(102.8, FH=2, weeks_to=2, weeks_from=2, bench_weight=[0.01, 0.1, 0.01, 0], use_90=True)
