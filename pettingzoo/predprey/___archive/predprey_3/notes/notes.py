FIX FIRST: step (base, 336): rename opponent_layer into evader_layer and generalize
# get the agent
agent = self.agent_list[self.agent_name_mapping[agent]]

# agent: agent instance
agent = self.agent_list[self.agent_name_mapping[self.agent_selection]]

rename self.predators in self.predators_list


# agent_list is a list of instances
# agents is s list of strings
self.agent_list = []
self.agents = []
self.dead_agents = []

for i in range(self.num_archers):
    name = "archer_" + str(i)
    print(f"archer{self.archer_player_num}")
    
    self.archer_dict[f"archer{self.archer_player_num}"] = Archer(
        agent_name=name
    )
    print(Archer(
        agent_name=name
    ))
    self.archer_dict[f"archer{self.archer_player_num}"].offset(i * 50, 0)
    self.archer_list.add(self.archer_dict[f"archer{self.archer_player_num}"])
    self.agent_list.append(self.archer_dict[f"archer{self.archer_player_num}"])
    if i != self.num_archers - 1:
        self.archer_player_num += 1



for i in range(self.num_knights):
    name = "knight_" + str(i)
    self.knight_dict[f"knight{self.knight_player_num}"] = Knight(
        agent_name=name
    )
    self.knight_dict[f"knight{self.knight_player_num}"].offset(i * 50, 0)
    self.knight_list.add(self.knight_dict[f"knight{self.knight_player_num}"])
    self.agent_list.append(self.knight_dict[f"knight{self.knight_player_num}"])
    if i != self.num_knights - 1:
        self.knight_player_num += 1

self.agent_name_mapping = {}
a_count = 0
for i in range(self.num_archers):
    a_name = "archer_" + str(i)
    self.agents.append(a_name)
    self.agent_name_mapping[a_name] = a_count
    a_count += 1
for i in range(self.num_knights):
    k_name = "knight_" + str(i)
    self.agents.append(k_name)
    self.agent_name_mapping[k_name] = a_count
    a_count += 1

