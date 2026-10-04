# PSO Path Planning

## Student Information

**Name:** Fatma Bashir  
**Roll Number:** 01-136232-063  
**Seed Used:** 63  

## Project Description

This project implements Particle Swarm Optimization (PSO) to find a short obstacle-free path from a start point to a goal point on a 2D grid.

The grid size, obstacles, start point, and goal point are generated programmatically using the roll number as the random seed. This makes the problem instance unique for each student.

## Approach

Each particle represents a possible path from the start point to the goal point using intermediate waypoints.

The fitness function evaluates each path based on:

- Total path distance
- Obstacle collisions

A large penalty is added whenever a path crosses an obstacle.

During each iteration, PSO updates the particles using:

- Personal Best (PBest)
- Global Best (GBest)
- Velocity
- Position

The process continues until the maximum number of iterations is reached.

The best obstacle-free path found by the swarm is then displayed on the grid.

## PSO Parameters

- Number of particles: 60
- Number of waypoints: 5
- Number of iterations: 400
- Inertia weight (w): 0.6
- Cognitive coefficient (c1): 1.7
- Social coefficient (c2): 1.7

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt