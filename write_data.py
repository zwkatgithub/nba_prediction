import pymysql
import json
from collections import defaultdict
import numpy 
import pandas
def connDatabase():
    conn = pymysql.connect(host='192.168.0.2',port=3306,user='team1',password='12345qwert')
    cursor = conn.cursor()
    cursor.execute('use zwk;')
    return conn, cursor

class PlayerData:
    __colName = ['fg_pct','three_pt_pct','fta','oreb_pct',
        'dreb_pct','ast_pct','tov','stl','blk','pf','p_m']
    def __init__(self, kwargs):
        self.player_id = kwargs['player_id']
        self.season = kwargs['season']
        self.__data = {
            colName:kwargs.get(colName) for colName in self.__colName
        }
    @property
    def data(self):
        return self.__data
    @staticmethod
    def colName():
        return PlayerData.__colName
        
def getTeamPlayer(cursor, game_id, team_id):
    sql = "SELECT player_id,start_pos FROM box_score WHERE game_id='{0}' and team_id={1} and start_pos!='';".format(game_id, team_id)
    #   print(sql)
    cursor.execute(sql)
    res = cursor.fetchall()
    if not res:
        return None
    assert len(res) == 5
    return [(row[0],row[1]) for row in res]
    #sql = "SELECT player_id FROM game_base_data WHERE game_base_data_id={0};".format(game_base_data_id)

def getPlayerData(cursor,player_id,season):
    data = {'player_id':player_id, 'season':season}
    sql = "SELECT ast_pct, oreb_pct, dreb_pct FROM player_advanced WHERE player_id={0} and season='{1}';"
    cursor.execute(sql.format(player_id,season))
    res = cursor.fetcPlayerDatahone()
    
    if res is None:
        return None
    
    data['ast_pct'], data['oreb_pct'], data['dreb_pct'] = res
    sql = "SELECT game_base_data_id FROM player_base WHERE player_id='{0}' and season={1};"
    cursor.execute(sql.format(player_id,season))
    res = cursor.fetchone()
    if res is None:
        return None
    #print(data)
    sql = "SELECT fg_pct, fg3_pct, fta, tov, stl, blk, pf, plus_minus FROM game_base_data WHERE game_base_data_id={0};"
    cursor.execute(sql.format(res[0]))
    res = cursor.fetchone()
    if res is None:
        return None
    #print(res)
    data['fg_pct'], data['three_pt_pct'], data['fta'], data['tov'], data['stl'],data['blk'],data['pf'],data['p_m'] = res
   
    return PlayerData(data).data

class OutputData:
    __colName = ['oreb','dreb','ast','stl','blk','in_pts','tov','ft','three_pt']

    def __init__(self, data):
        self.__data = {
            colName:data.get(colName) for colName in self.__colName
        }
    @property
    def data(self):
        return self.__data
    @staticmethod
    def colName():
        return OutputData.__colName

def getOutputData(cursor, game_id, team_id):
    sql = "SELECT oreb,dreb,ast,stl,blk,in_pts, tov,ft,three_pt FROM game_total where game_id='{0}' and team_id={1};".format(game_id,team_id)
    #print(sql)
    cursor.execute(sql)
    res = cursor.fetchone()
    if res is None:
        return None
    #data = {'game_id':game_id}
    #print(res)
    data = {}
    data['oreb'],data['dreb'],data['ast'],data['stl'],data['blk'],data['in_pts'],data['tov'],data['ft'],data['three_pt'] = res
    return data

def getGameData(cursor,season):
    sql = "SELECT game_id, home_team_id, vistor_team_id FROM game where season='{0}';".format(season)
    #print(sql)
    cursor.execute(sql)
    res = cursor.fetchall()
    for row in res:
        game_id,home_team_id, vistor_team_id = row
        yield game_id, home_team_id, vistor_team_id
def processTeam(cursor,players,season):
    data = defaultdict(list)
    for player in players:
        #print(player)
        data[player[1]].append(getPlayerData(cursor, player[0],season))
    return data
dataFolder = './data/raw/data-{0}.json'
labelFolder = './data/raw/label-{0}.json'

processedDataFolder = './data/processed/data.csv'


def playerDataMinus(hp, vp):
    data = []
    for colName in PlayerData.colName():
        data.append(float(hp[colName] - vp[colName]))
    return data

def processPlayerData(hd,vd):
    pos = ('F','C','G')
    data = []
    for p in pos:
        for d in zip(hd[p],vd[p]):
            res = playerDataMinus(d[0],d[1])
            data += res
    return data
def processLabel(l):
    data = []
    for colName in OutputData.colName():
        data.append(l[colName])
    return data

def write_data():
    res = []
    # colName = PlayerData.colName()+OutputData.colName()
    # res.append(str(colName).replace('\'','')[1:-1])
    for season in [str(n) for n in range(7,17)]:
        with open(dataFolder.format(season),'r') as f:
            data = json.load(f)
        with open(labelFolder.format(season),'r') as f:
            label = json.load(f)
        
        #print(colName)
        
        #print(data[0])
        for row in zip(data,label):
            d,l = row
            assert len(d) == 2
            assert len(l) == 2
            hd, vd = d
            hl, vl = l
            hdp = processPlayerData(hd,vd)
            hlp = processLabel(hl)
            res.append(str(hdp+hlp)[1:-1])
            vdp = processPlayerData(vd,hd)
            vlp = processLabel(vl)
            res.append(str(vdp+vlp)[1:-1])
    with open(processedDataFolder,'w') as f:
        #print(data[0])
        f.write('\n'.join(res))
def wirte_raw_data(season):
    
    data = []
    label = []
    conn, cursor = connDatabase()
    for game_id, home_team_id, vistor_team_id in getGameData(cursor, season):
        print(game_id)
        home_players = getTeamPlayer(cursor,game_id,home_team_id)
        vistor_players = getTeamPlayer(cursor,game_id,vistor_team_id)
        home_players_data = processTeam(cursor,home_players,season)
        vistor_player_data = processTeam(cursor,vistor_players,season)
        data.append([home_players_data,vistor_player_data])
        home_output = getOutputData(cursor,game_id,home_team_id)
        vistor_output = getOutputData(cursor, game_id,vistor_team_id)
        label.append([home_output, vistor_output])
    with open('./data/data-{0}.json'.format(season),'w') as f:
        json.dump(data,f)
    #with open('./data/label-{0}.json'.format(season),'w') as f:
    #    json.dump(label,f)
def normalize():
    dl = pandas.read_csv(processedDataFolder)
    data = dl.iloc[:,:-9].as_matrix()
    label = dl.iloc[:,-9:].as_matrix()
    norm_data = (data - data.mean(axis=0))/data.std(axis=0)
    
        

if __name__ == '__main__':
    # import argparse
    # import json
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-s','--season')
    # args = parser.parse_args()
    # season = args.season
    # for season in [str(n) for n in range(7,17)]:
    write_data()



