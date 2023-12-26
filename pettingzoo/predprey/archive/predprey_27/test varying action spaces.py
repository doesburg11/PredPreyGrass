import random

max_action_range = 7
max_action_offset = int((max_action_range - 1) / 2) 
action_range = 5
action_offset = int((action_range - 1) / 2) 
moore = False

print("max_action_offset ",max_action_offset)
print("action_offset ",action_offset)
action_range_array = list(range(-max_action_offset, max_action_offset+1))
#print("action_range_array ",action_range_array)


motion_range = []
row=0
for i in action_range_array:
    for j in action_range_array:
        motion_range.append([j,i])

print("motion_range",motion_range)
print()


action_space_array = []
row=0
for i in action_range_array:
    for j in action_range_array:
        item = motion_range[row]
        if abs(item[0]) < action_offset + 1 and abs(item[1]) < action_offset + 1:
            if moore:
                print(item)
                action_space_array.append(row)
            else:
                if abs(item[0]) + abs(item[1]) <= action_offset:
                    print(item)
                    action_space_array.append(row)
                     
        row+=1

print("action_space_array",action_space_array)
print("len(action_space_array) ",len(action_space_array))

sample_action = random.choice(action_space_array)

print("sample_action ", sample_action)




