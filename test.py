#!/usr/bin/env python

from MrWSI.core.problem import Problem
from MrWSI.core.platform import Context, Machine, Platform

if __name__ == "__main__":
    problem = Problem.load("./resources/workflows/CyberShake_30.wrk", "./resources/platforms/EC2.plt")
    context = Context(problem)
    machine = Machine(problem, context)
    machine.print_list()
    capacities = problem.type_capacities(0)
    demands = problem.task_demands(0)
    runtime = problem.task_runtime(0, 0)
    print(capacities, demands, runtime)
    st, node = machine.earliest_slot(capacities, demands, runtime, 0)
    print(st)
    item = machine.alloc_item(0, demands, runtime, node)
    machine.print_list()
    item = machine.alloc_item(1, demands, runtime)
    machine.print_list()
    item = machine.alloc_item(1, demands, runtime)
    machine.print_list()
    machine.extend_item(item, 0, 10)
    machine.print_list()
    print(machine.extendable_interval(item, capacities))
