#!/usr/bin/env python3
"""
Analyze carcass-sharing behavior: when predators leave carcasses (partial consumption),
do their own children or unrelated predators benefit?

This script analyzes agent_event_log JSON to determine:
1. Who created each carcass (first predator to bite a prey)
2. Who ate from that carcass afterward
3. Kinship relationships between predators
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_event_log(json_path):
    """Load agent event log from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def build_kinship_graph(logs):
    """Build parent-child relationships from logs."""
    children_of = defaultdict(list)  # parent_id -> [child_ids]
    parent_of = {}                    # child_id -> parent_id
    
    for agent_id, rec in logs.items():
        parent_id = rec.get('parent_id')
        if parent_id:
            parent_of[agent_id] = parent_id
            children_of[parent_id].append(agent_id)
    
    return children_of, parent_of


def is_offspring(agent_id, potential_parent_id, parent_of):
    """Check if agent_id is direct child of potential_parent_id."""
    return parent_of.get(agent_id) == potential_parent_id


def get_all_descendants(agent_id, children_of):
    """Get all descendants (children, grandchildren, etc.) of an agent."""
    descendants = set()
    to_visit = list(children_of.get(agent_id, []))
    
    while to_visit:
        child = to_visit.pop()
        if child not in descendants:
            descendants.add(child)
            to_visit.extend(children_of.get(child, []))
    
    return descendants


def analyze_carcass_sharing(logs):
    """
    Analyze who benefits from carcasses created by predators.
    
    Logic:
    - First predator to bite a prey (alive_before_bite=True) creates the carcass
    - Subsequent predators eating that prey (alive_before_bite=False) benefit from it
    """
    children_of, parent_of = build_kinship_graph(logs)
    
    # Track who created each carcass (prey_id -> first_predator_id)
    carcass_creators = {}
    
    # Build timeline of all eating events
    all_eating_events = []
    for predator_id, rec in logs.items():
        if 'predator' not in predator_id:
            continue
        
        for eating_event in rec.get('eating_events', []):
            all_eating_events.append({
                'predator_id': predator_id,
                'prey_id': eating_event['id_eaten'],
                't': eating_event['t'],
                'alive_before_bite': eating_event['alive_before_bite'],
                'bite_size': eating_event['bite_size'],
            })
    
    # Sort by time to process in order
    all_eating_events.sort(key=lambda e: (e['t'], e['alive_before_bite'] is False))
    
    # Identify carcass creators (first to bite each prey)
    for event in all_eating_events:
        prey_id = event['prey_id']
        if event['alive_before_bite'] and prey_id not in carcass_creators:
            carcass_creators[prey_id] = event['predator_id']
    
    # Analyze who benefits from carcasses
    stats = {
        'own_child': {'count': 0, 'energy': 0.0, 'events': []},
        'descendant': {'count': 0, 'energy': 0.0, 'events': []},
        'parent': {'count': 0, 'energy': 0.0, 'events': []},
        'sibling': {'count': 0, 'energy': 0.0, 'events': []},
        'unrelated': {'count': 0, 'energy': 0.0, 'events': []},
        'self_consumption': {'count': 0, 'energy': 0.0},  # Predator continues eating own carcass
    }
    
    for event in all_eating_events:
        if not event['alive_before_bite']:  # Eating from carcass
            prey_id = event['prey_id']
            eater_id = event['predator_id']
            creator_id = carcass_creators.get(prey_id)
            
            if not creator_id:
                continue  # No creator found (shouldn't happen)
            
            bite_size = event['bite_size']
            
            if eater_id == creator_id:
                # Same predator continues eating
                stats['self_consumption']['count'] += 1
                stats['self_consumption']['energy'] += bite_size
            else:
                # Different predator benefits
                event_record = {
                    't': event['t'],
                    'creator': creator_id,
                    'beneficiary': eater_id,
                    'prey': prey_id,
                    'energy': bite_size,
                }
                
                # Classify relationship
                if is_offspring(eater_id, creator_id, parent_of):
                    stats['own_child']['count'] += 1
                    stats['own_child']['energy'] += bite_size
                    stats['own_child']['events'].append(event_record)
                elif eater_id in get_all_descendants(creator_id, children_of):
                    stats['descendant']['count'] += 1
                    stats['descendant']['energy'] += bite_size
                    stats['descendant']['events'].append(event_record)
                elif is_offspring(creator_id, eater_id, parent_of):
                    stats['parent']['count'] += 1
                    stats['parent']['energy'] += bite_size
                    stats['parent']['events'].append(event_record)
                elif parent_of.get(eater_id) == parent_of.get(creator_id) and parent_of.get(eater_id):
                    stats['sibling']['count'] += 1
                    stats['sibling']['energy'] += bite_size
                    stats['sibling']['events'].append(event_record)
                else:
                    stats['unrelated']['count'] += 1
                    stats['unrelated']['energy'] += bite_size
                    stats['unrelated']['events'].append(event_record)
    
    return stats, carcass_creators


def print_results(stats):
    """Print analysis results."""
    print("=" * 70)
    print("CARCASS SHARING ANALYSIS")
    print("=" * 70)
    print("\nWhen predators leave carcasses (partial consumption), who benefits?\n")
    
    total_shared = (stats['own_child']['count'] + stats['descendant']['count'] + 
                    stats['parent']['count'] + stats['sibling']['count'] + 
                    stats['unrelated']['count'])
    
    total_energy_shared = (stats['own_child']['energy'] + stats['descendant']['energy'] + 
                          stats['parent']['energy'] + stats['sibling']['energy'] + 
                          stats['unrelated']['energy'])
    
    if total_shared == 0:
        print("No carcass sharing observed in this simulation.")
        return
    
    print(f"Total carcass-sharing events: {total_shared}")
    print(f"Total energy shared via carcasses: {total_energy_shared:.2f}\n")
    
    print("Breakdown by relationship:")
    print("-" * 70)
    
    categories = [
        ('Own children (direct offspring)', 'own_child'),
        ('Descendants (grandchildren+)', 'descendant'),
        ('Parent (reverse)', 'parent'),
        ('Siblings (same parent)', 'sibling'),
        ('Unrelated predators', 'unrelated'),
    ]
    
    for label, key in categories:
        count = stats[key]['count']
        energy = stats[key]['energy']
        pct = (count / total_shared * 100) if total_shared > 0 else 0
        energy_pct = (energy / total_energy_shared * 100) if total_energy_shared > 0 else 0
        
        print(f"{label:.<40} {count:>4} ({pct:>5.1f}%)  |  {energy:>7.2f} energy ({energy_pct:>5.1f}%)")
    
    print(f"\n{'Self-consumption (same predator)':.<40} {stats['self_consumption']['count']:>4}  |  "
          f"{stats['self_consumption']['energy']:>7.2f} energy")
    
    print("\n" + "=" * 70)
    print("KIN SELECTION METRICS")
    print("=" * 70 + "\n")
    
    # Kin bias ratio
    kin_count = stats['own_child']['count'] + stats['descendant']['count']
    kin_energy = stats['own_child']['energy'] + stats['descendant']['energy']
    unrelated_count = stats['unrelated']['count']
    unrelated_energy = stats['unrelated']['energy']
    
    if unrelated_count > 0:
        kin_bias_ratio = kin_count / unrelated_count
        energy_bias_ratio = kin_energy / unrelated_energy if unrelated_energy > 0 else float('inf')
        
        print(f"Kin bias ratio (count):  {kin_bias_ratio:.2f}x")
        print(f"Kin bias ratio (energy): {energy_bias_ratio:.2f}x")
        print("\n> 1.0 = Own descendants benefit MORE than unrelated predators")
        print("< 1.0 = Unrelated predators benefit MORE than own descendants")
        print("= 1.0 = No bias (random/neutral sharing)")
        
        if kin_bias_ratio > 1.5:
            print("\n✓ Strong kin favoritism detected!")
        elif kin_bias_ratio > 1.1:
            print("\n✓ Moderate kin favoritism detected")
        elif kin_bias_ratio < 0.9:
            print("\n✗ Kin avoidance or competitive exclusion")
        else:
            print("\n≈ Neutral/random sharing pattern")
    else:
        print("No sharing with unrelated predators observed.")
        if kin_count > 0:
            print("✓ All carcass sharing was with kin!")
    
    # Show example events
    print("\n" + "=" * 70)
    print("EXAMPLE EVENTS (first 5 of each type)")
    print("=" * 70 + "\n")
    
    for label, key in categories[:3]:  # Show examples for top 3 categories
        if stats[key]['events']:
            print(f"\n{label}:")
            for event in stats[key]['events'][:5]:
                print(f"  t={event['t']:>4}: {event['creator']} → {event['beneficiary']} "
                      f"(prey: {event['prey']}, energy: {event['energy']:.2f})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_carcass_sharing.py <path_to_agent_event_log.json>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    print(f"Loading event log from: {json_path}")
    logs = load_event_log(json_path)
    
    print(f"Loaded {len(logs)} agents")
    stats, carcass_creators = analyze_carcass_sharing(logs)
    
    print(f"Identified {len(carcass_creators)} carcasses created\n")
    
    print_results(stats)


if __name__ == '__main__':
    main()
