    def print_model_state_to_screen(self, agent_type):
        # Determine the maximum width needed for the values
        max_width = max(
            len(f"{value:.1f}") for row in self.model_state[agent_type] for value in row
        )
        transposed_matrix = zip(*self.model_state[agent_type])
        for row in transposed_matrix:
            formatted_row = "  ".join(
                f"{value:>{max_width}.1f}" if value != 0 else "." * max_width
                for value in row
            )
            print(f"[  {formatted_row}  ]")

    def print_agent_instance_in_grid_location(self, agent_type_nr):
        for y in range(self.x_grid_size):
            print("[", end="  ")
            for x in range(self.y_grid_size):
                agent_instance = self.agent_instance_in_grid_location[agent_type_nr, x, y]
                if agent_instance is None:
                    print("...", end="  ")
                else:
                    print(".", end="")
                    print(agent_instance.agent_name.split("_")[-1], end="  ")
            print("]")

    def debug_check(self, agent_type_nr):
        print("\n" + "=" * 85)
        agent_type_name = self.agent_type_name_list[agent_type_nr]
        print(f"START CHECKING {agent_type_name.upper()}:\nself.n_cycles: {self.n_cycles}")
        if agent_type_nr == self.predator_type_nr:
            print(f"self.n_active_predator: {self.n_active_predator}")
        elif agent_type_nr == self.prey_type_nr:
            print(f"self.n_active_prey: {self.n_active_prey}")
        elif agent_type_nr == self.grass_type_nr:
            print(f"self.n_active_grass: {self.n_active_grass}")
        print("self.print_model_state_to_screen")
        self.print_model_state_to_screen(agent_type_nr)
        print("\nself.agent_instances_in_grid_location:")
        self.print_agent_instance_in_grid_location(agent_type_nr)
        print(f"\nEND CHECKING\n{'=' * 85}")
