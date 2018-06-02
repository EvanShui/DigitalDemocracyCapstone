
# coding: utf-8

# # Upleveled Data Processing

# In[ ]:


import sys
import numpy as np
import pandas as pd


# ### Define constants.  Determine the upleveled transcript that will be processed.

# In[ ]:


default_original_upleveled_filename = "../data/original/upleveled.txt"
default_temp_folder = "."


# In[ ]:


original_upleveled_filename = default_original_upleveled_filename
temp_folder = default_temp_folder


# In[ ]:


#constants
TEMP_ORIGINAL_UPLEVELED_SORTED_FILENAME = temp_folder + "/temp_original_upleveled_sorted.csv"
TEMP_MARKED_UPLEVELED_FILENAME = temp_folder + "/temp_marked_upleveled.csv"
TEMP_BILL_START_END_TIMES_ALL_FILENAME = temp_folder + "/temp_bill_start_end_times_all.csv"
TEMP_BILL_START_END_TIMES_LONGEST_FILENAME = temp_folder + "/temp_bill_start_end_times_longest.csv"


# ### Read in the transcript specified above, and sort by video_id, hearing_id, and speaker_start_time.

# In[ ]:


print("...Creating bill time tables...")


# In[ ]:


upleveled = pd.read_table(original_upleveled_filename, sep='~~~~~', engine='python')
upleveled = upleveled.sort_values(["video_id", "hearing_id", "speaker_start_time"])
upleveled.to_csv(TEMP_ORIGINAL_UPLEVELED_SORTED_FILENAME, sep="~", index=False)


# ### Tag lines with bill_change_tag, which increments every time the bill is changed.

# In[ ]:


def tag_bill_change_lines(original, cleaned):
    line = original.readline()
    current_bill_id = line.split("~")[0]
    i = 0
    cleaned.write(line.rstrip("\n") + "~0\n")
    
    for line in original:
        line_splits = line.split("~")
        
        if (line_splits[0] != current_bill_id):
            current_bill_id = line_splits[0]
            i += 1
        
        cleaned.write(line.rstrip("\n") + "~" + str(i) + "\n")


# In[ ]:


with open(TEMP_ORIGINAL_UPLEVELED_SORTED_FILENAME, 'r') as original:
    with open(TEMP_MARKED_UPLEVELED_FILENAME, 'w') as cleaned:
        #consume/write headings
        h = original.readline()
        cleaned.write(h.rstrip("\n") + "~bill_change_tag\n")
            
        tag_bill_change_lines(original, cleaned)


# ### Create tables of bills and when they are being discussed.

# In[ ]:


#tagged_upleveled = pd.read_table(TEMP_MARKED_UPLEVELED_FILENAME, sep='~')


# In[ ]:


tagged_upleveled = pd.read_table(TEMP_MARKED_UPLEVELED_FILENAME, sep='~')
video_ids = np.unique(tagged_upleveled['video_id'])[1:11:3]
tagged_upleveled = tagged_upleveled[~tagged_upleveled['video_id'].isin(video_ids)]


# In[ ]:


bill_start_times = tagged_upleveled.groupby(["bill_change_tag"]).head(1)
bill_end_times = tagged_upleveled.groupby(["bill_change_tag"]).tail(1)
bill_start_end_times = pd.merge(bill_start_times[["bill_id", "hearing_id", "video_id", "speaker_start_time", "bill_change_tag"]],
                                bill_end_times[["speaker_end_time", "bill_change_tag"]],
                                on=["bill_change_tag"]).drop(["bill_change_tag"], axis=1)
bill_start_end_times["length"] = bill_start_end_times["speaker_end_time"] - bill_start_end_times["speaker_start_time"]
bill_start_end_times = bill_start_end_times.sort_values(["video_id", "speaker_start_time"])


# In[ ]:


longest_bill_discussions = bill_start_end_times.sort_values(["bill_id", "length"]).groupby(["bill_id"]).tail(1)
longest_bill_discussions = longest_bill_discussions.sort_values(["video_id", "speaker_start_time"])


# In[ ]:


bill_start_end_times.to_csv(TEMP_BILL_START_END_TIMES_ALL_FILENAME, sep="~", index=False)
longest_bill_discussions.to_csv(TEMP_BILL_START_END_TIMES_LONGEST_FILENAME, sep="~", index=False)
print("...Creating bill time tables complete.")

