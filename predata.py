import pymysql

def connDatabase():
    conn = pymysql.connect(host='192.168.0.2',port=3306,user='team1',password='12345qwert')
    cursor = conn.cursor()
    cursor.fet
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
        
def getTeamPlayer(cursor, game_id, team_id):
    sql = "SELECT player_id,start_pos FROM box_score WHERE game_id='{0}',team_id={1}, start_pos!='';".format(game_id, team_id)
    cursor.execute(sql)
    res = cursor.fetchall()
    if not res:
        return None
    assert len(res) == 5
    return [(row[0],row[1]) for row in res]
    #sql = "SELECT player_id FROM game_base_data WHERE game_base_data_id={0};".format(game_base_data_id)

def getPlayerData(cursor,player_id,season):
    data = {'player_id':player_id, 'season':season}
    sql = "SELECT ast_pct, oreb_pct, dreb_pct FROM player_advanced WHERE player_id={0},season='{1}';"
    cursor.execute(sql.format(player_id,season))
    res = cursor.fetchone()
    if res is None:
        return None
    data['ast_pct'], data['oreb_pct'], data['dreb_pct'] = res
    sql = "SELECT game_base_data_id FROM player_base WHERE player_id={0}, season='{1}';"
    cursor.execute(sql.format(player_id,season))
    res = cursor.fetchone()
    if res is None:
        return None
    sql = "SELECT fg_pct, fg3_pct, fta, tov, stl, blk, pf, plus_minus FROM game_base_data WHERE game_base_data_id={0};"
    cursor.execute(sql.format(res[0]))
    res = cursor.fetchone()
    if res is None:
        return None
    data['fg_pct'], data['fg3_pct'], data['fta'], data['tov'], data['stl'],data['blk'],data['pf'],data['p_m'] = res
    return PlayerData(data)

class OutputData:
    __colName = ['game_id','team_id','oreb','dreb','ast','stl','blk','in_pts','tov','ft','three_pt']

    def __init__(self, data):
        self.__data = {
            colName:data.get(colName) for colName in self.__colName
        }
    @property
    def data(self):
        return self.__data

def getOutputData(cursor, game_id):
    output = []
    sql = "SELECT * FROM game_total where game_id='{0}'".format(game_id)
    cursor.execute(sql)
    res = cursor.ftechall()
    for row in res:
        data = {'game_id':game_id}
        data['team_id'],data['oreb'],data['dreb'],data['ast'],data['stl'],data['blk'],data['in_pts'],data['tov'],data['ft'],data['three_pt'] = row[2:]
        output.append(OutputData(data))
    return output

def getGameData(cursor,season):
    sql = "SELECT game_id, home_team_id, vistor_team_id FROM game where season='{0}';".format(season)
    #print(sql)
    cursor.execute(sql)
    res = cursor.fetchall()
    for row in res:
        game_id,home_team_id, vistor_team_id = row
        yield game_id, home_team_id, vistor_team_id
def processTeam(players):
    data = {}
    for player in players:
        data[player[1]] = getPlayerData(cursor, player[0],season)
    return data
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--season')
    args = parser.parse_args()

    season = args.season
    data = []
    label = []
    conn, cursor = connDatabase()
    for game_id, home_team_id, vistor_team_id in getGameData(cursor, season):
        home_players = getTeamPlayer(cursor,game_id,home_team_id)
        vistor_players = getTeamPlayer(cursor,game_id,vistor_team_id)
        for player_id in home_team_id:
            player = getPlayerData(cursor,player_id,season)

