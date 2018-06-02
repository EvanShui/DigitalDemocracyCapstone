import MySQLdb

'''
Will access dev server to grab all the utterances from the Utterance table, sort by 
the video number, and append all text to a text file based off of video number. 
Name of file will have the following convention: 'vid_trasncript.txt' where vid is 
the video id number.
'''

sql_query = "SELECT * FROM Utterance WHERE vid=%d LIMIT 100"

def exec_query(query):
    cur.execute(query) 
    return cur.fetchall()

def get_transcript():

    db = MySQLdb.connect(host="127.0.0.1",
            user="root",
            passwd="",
            db="DDDB2016Aug")

    cur = db.cursor()
    utterance_query = "SELECT time, endTime, text FROM Utterance where vid=1;"
    transcript = []
    val = 0

    cur.execute(utterance_query)
    result = cur.fetchall()
    #grabs each distince VID and append to a list 
    for line in result: 
        transcript.append({'start':line[0], 'end':line[1], 'text': line[2], 'video_id': 1})
    return(transcript)

def main():
    trans = get_transcript()
    print(trans)

main()
