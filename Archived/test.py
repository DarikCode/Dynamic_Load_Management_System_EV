from Archived.dashboard import get_battery_physics_rate


def smart_dlm_optimizer(
        grid_limit, solar_kw, price, vehicles, price_threshold=0.20):
    total_capacity = grid_limit + solar_kw
    allocations = {v['id']: (0.0, "Waiting/Idle") for v in vehicles}
    active_evs = [
        v for v in vehicles if v['active'] and v['soc'] < v['target']
    ]

    if not active_evs:
        return allocations, 0.0

    remaining_power = total_capacity
    total_allocated = 0.0
    high_price_mode = price > price_threshold
    priority_map = {'Critical': 3, 'VIP': 2, 'Regular': 1}
    active_evs.sort(key=lambda x: priority_map[x['type']], reverse=True)

    for v in active_evs:
        phys_max = get_battery_physics_rate(v['soc'], v['max_kw'])
        if v['type'] == 'Regular' and high_price_mode and solar_kw < 10:
            econ_max = 2.0
            status_note = "💰 Econ Throttled"
        else:
            econ_max = phys_max
            status_note = "⚡ Fast Charging"

        wanted = min(phys_max, econ_max)
        given = min(wanted, remaining_power)
        allocations[v['id']] = (given, status_note)
        remaining_power -= given
        total_allocated += given

    return allocations, total_allocated