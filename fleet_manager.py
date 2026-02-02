import random

# 1. Database of Real EV Specifications
EV_MODELS = [
    {'model': 'Tesla Model 3', 'cap': 75, 'kw': 11},
    {'model': 'Tesla Model Y', 'cap': 82, 'kw': 11},
    {'model': 'Nissan Leaf', 'cap': 40, 'kw': 6.6},
    {'model': 'Ford F-150 Lightning', 'cap': 131, 'kw': 19.2},
    {'model': 'Rivian R1T', 'cap': 135, 'kw': 22},
    {'model': 'Hyundai Ioniq 5', 'cap': 77, 'kw': 11},
    {'model': 'Kia EV6', 'cap': 77, 'kw': 11},
    {'model': 'Porsche Taycan', 'cap': 93, 'kw': 22},
    {'model': 'Lucid Air', 'cap': 112, 'kw': 19},
    {'model': 'Chevy Bolt', 'cap': 65, 'kw': 7.2},
    {'model': 'BMW iX', 'cap': 105, 'kw': 11},
    {'model': 'Audi e-tron', 'cap': 95, 'kw': 11},
    {'model': 'VW ID.4', 'cap': 82, 'kw': 11}
]

# 2. Configuration
TYPES = ['Regular', 'VIP', 'Critical']
TYPE_WEIGHTS = [0.7, 0.2, 0.1]


# [CHANGE] Default set to 50
def generate_fleet(num_cars=50): # Generates a list of N random vehicles with realistic attributes.

    fleet = []

    for i in range(num_cars):
        template = random.choice(EV_MODELS)
        car_type = random.choices(TYPES, weights=TYPE_WEIGHTS, k=1)[0]

        # Profile Logic
        if car_type == 'Critical':
            start_soc = random.randint(5, 30)
            target_soc = 100
        elif car_type == 'VIP':
            start_soc = random.randint(20, 50)
            target_soc = 90
        else:  # Regular
            start_soc = random.randint(30, 70)
            target_soc = 80

        soh = random.uniform(85.0, 100.0)
        real_capacity = template['cap'] * (soh / 100.0)

        car = {
            'id': f"EV-{1000 + i}",
            'model': template['model'],
            'type': car_type,
            'cap_kwh': real_capacity,
            'max_kw': template['kw'],
            'soc': start_soc,
            'target_soc': target_soc,
            'soh': soh,
            'status': 'Idle',
            'connected': True,
            'shadow_soc': start_soc,
            'current_kw': 0.0
        }
        fleet.append(car)

    return fleet