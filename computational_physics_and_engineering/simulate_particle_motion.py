import time
import numba


def simulate_particle_motion_without_numba(mass, initial_position, initial_velocity, constant_force, time_step, num_steps):
    position = initial_position
    velocity = initial_velocity

    for _ in range(num_steps):
        acceleration = constant_force / mass
        velocity += acceleration * time_step
        position += velocity * time_step

    return position


@numba.jit
def simulate_particle_motion_with_numba(mass, initial_position, initial_velocity, constant_force, time_step, num_steps):
    position = initial_position
    velocity = initial_velocity

    for _ in range(num_steps):
        acceleration = constant_force / mass
        velocity += acceleration * time_step
        position += velocity * time_step

    return position


def main():
    # Particle parameters
    mass = 1.0
    initial_position = 0.0
    initial_velocity = 0.0
    constant_force = 10.0

    # Simulation parameters
    time_step = 0.01
    num_steps = 10000000

    # Without Numba
    start_time = time.time()
    final_position_without_numba = simulate_particle_motion_without_numba(mass, initial_position, initial_velocity, constant_force, time_step, num_steps)
    end_time = time.time()
    execution_time_without_numba = end_time - start_time

    print("Simulation without Numba:")
    print("Final Position:", final_position_without_numba)
    print("Execution time:", execution_time_without_numba, "seconds")

    # With Numba
    start_time = time.time()
    final_position_with_numba = simulate_particle_motion_with_numba(mass, initial_position, initial_velocity, constant_force, time_step, num_steps)
    end_time = time.time()
    execution_time_with_numba = end_time - start_time

    print("Simulation with Numba:")
    print("Final Position:", final_position_with_numba)
    print("Execution time:", execution_time_with_numba, "seconds")


if __name__ == "__main__":
    main()
