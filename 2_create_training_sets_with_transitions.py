
# coding: utf-8

# In[ ]:

import pandas as pd


# In[ ]:

#constants
training_output_binary_filename = "UNDEFINED"
training_output_n_range_filename = "UNDEFINED"
training_output_n_range_collapsed_filename = "UNDEFINED"

#configurable values
cleaned_raw_filename = "UNDEFINED"
bill_start_end_times_filename = "UNDEFINED"

with open("CONSTANTS") as constants_file:
    for line in constants_file:
        line_splits = line.rstrip("\n").split("=")
        
        if (line_splits[0] == "TRAINING_BINARY"):
            training_output_binary_filename = line_splits[1]
        elif (line_splits[0] == "TRAINING_N_RANGE"):
            training_output_n_range_filename = line_splits[1]
        elif (line_splits[0] == "TRAINING_N_RANGE_COLLAPSED"):
            training_output_n_range_collapsed_filename = line_splits[1]
            
with open("CONFIG") as config_file:
    for line in config_file:
        line_splits= line.rstrip("\n").split("=")
        
        if (line_splits[0] == "WHICH_CLEANED_RAW"):
            cleaned_raw_filename = line_splits[1]
        elif (line_splits[0] == "WHICH_BILL_START_END_TIMES"):
            bill_start_end_times_filename = line_splits[1]


# # Mark Transition Lines (Binary)

# In[ ]:

def get_transition_value(line, bill_start_time, bill_end_time):
    utterance_start_time = int(line.split("~")[0])
    utterance_end_time = int(line.split("~")[1])
    
    if (utterance_end_time > bill_start_time and bill_end_time > utterance_start_time):
        return 1
    else:
        return 0


# In[ ]:

def mark_transition_lines(raw, bill_times, out):
    bill_times_splits = bill_times.readline().split("~")
    for line in raw:
        transition_value = get_transition_value(line, int(bill_times_splits[3]), int(bill_times_splits[4]))
        if (transition_value == 1):
            bill_line = bill_times.readline()
            if (bill_line == ""):
                bill_times_splits = [-1, -1, -1, -1, -1, -1]
            else:
                bill_times_splits = bill_line.split("~")
            
        out.write(line.rstrip('\n') + "~" + str(transition_value) + "\n")


# In[ ]:

with open(cleaned_raw_filename, 'r') as raw:
    with open(bill_start_end_times_filename, 'r') as bill_times:
        with open(training_output_binary_filename, 'w') as out:
            #consume/write headings
            raw.readline()
            bill_times.readline()
            out.write("start~end~text~video_id~transition_value\n")
            
            #mark the transitions
            mark_transition_lines(raw, bill_times, out)


# # Mark Transition Lines (Tertiary)

# # Mark Transition Lines (N Range)

# In[ ]:

n = 5


# In[ ]:

n_range = pd.read_csv(training_output_binary_filename, sep="~")


# In[ ]:

transition_indexes = n_range.index[n_range["transition_value"] == 1].tolist()
new_transition_indexes = []

length = len(n_range)
for i in transition_indexes:
    for x in range(-n, n+1):
        if (i + x >= 0 and i + x < length):
            new_transition_indexes.append(i + x)
            
n_range.loc[new_transition_indexes, "transition_value"] = 1

n_range.to_csv(training_output_n_range_filename, sep="~", index=False)


# # Collapse Transition Lines (N Range)

# In[ ]:

def collapse_transitions(uncollapsed, collapsed):
    accumulated_text = ""
    accumulating = False
    
    for line in uncollapsed:
        split_line = line.split("~")
        transition_value = int(split_line[4])
        text = split_line[2] + " "
        
        if transition_value == 1 and accumulating:
            accumulated_text = accumulated_text + text
        elif transition_value == 1 and not accumulating:
            accumulating = True
            accumulated_text = accumulated_text + text
        elif transition_value == 0 and accumulating:
            collapsed.write(split_line[0] + "~" + split_line[1] + "~" +
                            accumulated_text + "~" + split_line[3] + "~1\n")
            collapsed.write(line)
            accumulating = False
            accumulated_text = ""
        else:
            collapsed.write(line)


# In[ ]:

with open(training_output_n_range_filename, 'r') as uncollapsed:
    with open(training_output_n_range_collapsed_filename, 'w') as collapsed:
        #consume/write headings
        h = uncollapsed.readline()
        collapsed.write(h)
            
        #collapse transitions
        collapse_transitions(uncollapsed, collapsed)


# # Verification

# In[ ]:

binary = pd.read_csv(training_output_binary_filename, sep="~")
bill_times = pd.read_csv(bill_start_end_times_filename, sep="~")


# In[ ]:

assert(len(binary[binary["transition_value"]==1])==len(bill_times))
