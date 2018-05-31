#todo: import predict_transitions script to this
#create new transcript data that can take in list of dictionaries and push to TT_BillDiscussion
#table

import MySQLdb 
from DAOManager import DAOManager
import sys

dao_manager = DAOManager.create()

'''
from pull_utterance_capstone import get_transcript
from Predict_Transitions import predict, enhance_dictionary
'''

class model_wrapper: 
    def __init__(self, model_path, vector_path): 
        self.model_path = model_path
        self.vector_path = vector_path
        #these two files will be used in the predict_from_naive_bayes function
        #self.model = pickle.load(open(self.model_path, "rb"))
        self.model = (open(self.model_path, "rb"))
        self.vector = (open(self.vector_path, "rb"))

    def __str__(self):
        ret_str = ("Model path: {} Vector Path: {}".format(self.model_path,
                self.vector_path))
        return ret_str

    def __repr__(self):
        return str(self)

db = MySQLdb.connect(host="127.0.0.1",
        user="root",
        passwd="",
        db="DDDB2016Aug")

cur = db.cursor()

def find_billID(conn, bill_type, bill_num):

    '''
    Bill Discussion entry (dct) -> Bill Discussion entry (dct) 
    appends bill id to the bil ldictionary entry 
    ''' 
    try: 
        result = dao_manager.run_query(conn, SQL_bid_base % (bill_type,
            bill_num))
        bid = result[0][0]
    except: 
        bid = None 
    return bid

def main():
    #TODO: 
    #1. Run the capstone project program and see what output it and get
    #whatever other necessary info to push into TT_BillDiscussion table 
    #2. Implement saving the pickle file in memory and call capstone functions
    #using the pickle files saved in memory and alter scripts if necessary
    '''
    new_transcript = [{'start':1, 'end':9, 'text':'we are starting', 'video_id':1}, 
     {'start':9, 'end':12, 'text':'we keep going on assembly bill three four five six', 'video_id':1},
     {'start':14, 'end':20, 'text':'is this a', 'video_id':1},
     {'start':21, 'end':22, 'text':'what do you think of senate bill 134?', 'video_id':1}]
    '''
    
    model_wrap = model_wrapper("1.txt", "2.txt")
    print(model_wrap.model.read())
    '''
    example of how to use dao manager 

    with dao_manager.get_dao() as dao: 
        conn = dao.get_dddb_connection() 
        billId = find_billID(conn, 'A', 1000)
        print(billId)
    '''
    trans = get_transcript()
    transition_dictionary = predict(trans)
    enhanced_dictionary = enhance_dictionary(trans, transition_dictionary)
    for entry in enhanced_dictionary: 
        print(entry)
    close(model_wrap.model) 
    close(model_wrap.vector)
    '''
    close(model_wrap.model)
    close(model_wrap.vector)

if __name__ == "__main__":
    main()
    #trans = get_transcript()

