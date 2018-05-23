import pymysql
import requests

conn1 = pymysql.connect(host="192.168.0.2",port=3306,user='team1',password='12345qwert')
cursor_nba = conn1.cursor()
cursor_nba.execute('use nba;')

'''
nba find player
is in zwk
in : pass
not in : crawl
'''
url = 'http://stats.nba.com/stats/playerdashboardbyyearoveryear'
params = {
    'DateFrom':"", 
    'DateTo':"", 
    "GameSegment":"", 
    'LastNGames': '0',
    'LeagueID': '00',
    'Location':'', 
    'MeasureType': None,
    'Month': '0',
    'OpponentTeamID': '0',
    'Outcome':'', 
    'PORound': '0',
    'PaceAdjust': 'N',
    'PerMode': 'PerGame',
    'Period': '0',
    'PlayerID': None,
    'PlusMinus': 'N',
    'Rank': 'N',
    'Season': '2017-18', #2017-18
    'SeasonSegment':'', 
    'SeasonType': 'Regular Season',
    'ShotClockRange':'', 
    'Split': 'yoy',
    'VsConference': '',
    'VsDivision': ''
}
#'fg_pct 12','three_pt_pct 15','fta 17','oreb_pct=16',
#        'dreb_pct=17','ast_pct=15','tov 23','stl 24','blk 25','pf 27','p_m 30'
colNames = ['fg_pct','three_pt_pct','fta','oreb_pct','dreb_pct','ast_pct','tov','stl','blk','pf','p_m']

def crawl(player_id):
    params['PlayerID'] = player_id
    params['MeasureType'] = 'Base'
    base_data = requests.get(url, params=params, headers={'User-Agent':"Mozilla/5.0 (Linux; Android 5.1.1; Nexus 6 Build/LYZ28E) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.23 Mobile Safari/537.36"}).json()
    #dictt = genDict(data['resultSets'][1]['headers'])
    print("base_data")
    row = base_data['resultSets'][1]['rowSet'][0]
    fg_pct, three_pt_pct,fta ,tov, stl, blk, pf, p_m = row[12],row[15], row[17], row[23], row[24],row[25], row[27], row[28]
    params['MeasureType'] = 'Advanced'
    adv_data = requests.get(url, params=params, headers={'User-Agent':"Mozilla/5.0 (Linux; Android 5.1.1; Nexus 6 Build/LYZ28E) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.23 Mobile Safari/537.36"}).json()
    row = adv_data['resultSets'][1]['rowSet'][0]
    oreb_pct, dreb_pct, ast_pct = row[16], row[17], row[15]
    return [fg_pct, three_pt_pct, fta, oreb_pct, dreb_pct, ast_pct, tov, stl, blk, pf, p_m]
def write_database(player_id, data):
    sql = "UPDATE input_data SET "+'={}, '.join(colNames)+' WHERE player_base_id = {};'
    sql.format(colNames+[player_id])
    cursor_nba.execute(sql)


cursor_nba.execute('SELECT id FROM player_base WHERE id > 10;')
player_ids = [row[0] for row in cursor_nba.fetchall()]

for player_id in player_ids:
    print(player_id)
    data = crawl(player_id)
    write_database(player_id, data)

conn1.commit()


    

