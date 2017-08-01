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

    return round(cores, 2), ceil(memory)


def read_dax(dax_name):
    xml_file = "resources/workflows/{}.xml".format(dax_name)
    soup = BeautifulSoup(open(xml_file), "html.parser")

    with open("resources/workflows/Characteristics.json") as f:
        app_name = os.path.basename(dax_name).split("_")[0]
        chars = json.load(f)[app_name]

    tasks = {}
    for job in soup.find_all("job"):
        tasks[job["id"]] = {
            "runtime": ceil(abs(float(job["runtime"]))),
            "demands": generate_task_demand(chars[job["name"]]),
            "prevs": set(),
            "nexts": set()
        }

    for sec in soup.find_all("child"):
        child = sec["ref"]
        for p in sec.find_all("parent"):
            parent = p["ref"]
            tasks[child]["prevs"].add(parent)
            tasks[parent]["nexts"].add(child)

    for task in tasks.values():
        task["prevs"] = list(task["prevs"])
        task["nexts"] = list(task["nexts"])
    return tasks

def trans_dax_workflow(dax_name, path="resources/workflows"):
    tasks = read_dax(dax_name)
    filename = os.path.abspath(os.path.join(path, "{}.wrk".format(dax_name)))
    with open(filename, "w") as f:
        json.dump(tasks, f, indent=2)

if __name__ == '__main__':
    print(read_dax("Montage_50"))
