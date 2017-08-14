from bs4 import BeautifulSoup
from math import ceil
import simplejson as json
import os.path
from random import gauss


def generate_task_demand(chars):
    cores = gauss(chars["cores"]["mean"], chars["cores"]["std"]) / 100
    memory = gauss(chars["memory"]["mean"], chars["memory"]["std"])

    if cores <= 0:
        cores = 1

    if memory < 0:
        memory = 0

    return round(cores, 2), ceil(memory), 0


def read_dax(dax_path):
    xml_file = os.path.abspath(dax_path)
    soup = BeautifulSoup(open(xml_file), "html.parser")

    with open(
            os.path.join(os.path.split(xml_file)[0],
                         "Characteristics.json")) as f:
        app_name = os.path.basename(dax_path).split("_")[0]
        chars = json.load(f)[app_name]

    tasks = {}
    files = {}
    for job in soup.find_all("job"):
        tasks[job["id"]] = {
            "runtime": ceil(abs(float(job["runtime"]))),
            "demands": generate_task_demand(chars[job["name"]]),
            "prevs": {},
        }
        for uses in job.find_all("uses"):
            file_name = uses["file"]
            if file_name not in files:
                files[file_name] = [None, [], abs(int(uses["size"]))]
            if uses["link"] == "input":
                files[file_name][1].append(job["id"])
            else:
                files[file_name][0] = job["id"]

    for sec in soup.find_all("child"):
        child = sec["ref"]
        for p in sec.find_all("parent"):
            parent = p["ref"]
            tasks[child]["prevs"][parent] = 0

    for task_from, task_to_set, data in files.values():
        if task_from and task_to_set:
            for task_to in task_to_set:
                if task_from not in tasks[task_to]["prevs"]:
                    tasks[task_to]["prevs"][task_from] = data
                else:
                    tasks[task_to]["prevs"][task_from] += data

    return tasks


if __name__ == '__main__':
    from sys import argv
    tasks = read_dax(argv[1])
    with open(argv[1][:-4] + ".wrk", "w") as f:
        json.dump(tasks, f, indent=2)
