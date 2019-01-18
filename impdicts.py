from typing import Tuple, List, Dict

slot2tag = {'to_loc' : '$ACITY$','froms_loc' : '$DCITY$',
            'date' : '$DATE$','time' : '$TIME$',
            'class' : '$CLASS$', 'round_trip': '$RTRIP$' ,
            'city_name' : '$CITY$', 'transport_type' :'$TTYPE$'}

tag2slot = {'$ACITY$' : 'to_loc','$DCITY$': 'from_loc',
            '$DATE$' : 'date','$TIME$': 'time',
           '$CLASS$':  'class', '$RTRIP$': 'round_trip' ,
           '$CITY$' : 'city_name', '$TTYPE$' : 'transport_type' }


all_tags = ['$ACITY$','$DATE$','$CLASS$','$CITY$','$RTRIP$','$TTYPE$',
            '$DCITY$','$TIME$']


intent_tags = {
      'flight' : ['$ACITY$','$DATE$','$CLASS$','$DCITY$','$TIME$' ],
      'airfare' : ['$ACITY$','$DATE$','$CLASS$','$RTRIP$',
            '$DCITY$'],
      'airline' : ['$ACITY$','$CLASS$','$DCITY$'],
      'ground_service' : ['$CITY$', '$TTYPE$'],
      'ground_fare' : ['$CITY$', '$TTYPE$']
}

intent2indx =  {'flight': 0,'airfare' : 1,'airline' :2, 'ground_service' : 3, 'ground_fare' : 4}


indx2intent = {0:'flight',1:'airfare',2:'airline',
               3: 'ground_service', 4:'ground_fare'}
               

# Dictionaries
## DIctionary which maps intent to action space
intent2action ={ 0:[0, 1, 2, 3, 4, 8, 9, 11, 12, 13, 14, 15, 19], # Intent Flight
                    1:[ 0, 1, 3, 4, 5, 8, 11, 12, 14, 15, 16, 19], # Intent Airfare
                    2:[ 0, 1, 4, 8, 11, 12, 15, 19], # Intent airline
                    3:[ 6, 7, 10, 17, 18, 19], # Intent Ground Service
                    4:[ 6, 7, 10, 17, 18, 19] # Intent Ground Fare
}

# mapping of the intent to there relevant slots values
intent2slots = {0 : [0,1,2,3,4],
                1 : [0, 1, 3, 4, 5],
                2 : [0, 1, 4,],
                3 : [ 6, 7,],
                4 : [ 6, 7,],
                }


