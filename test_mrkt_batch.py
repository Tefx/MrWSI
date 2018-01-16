from test import *
import logging


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from mrkt.cluster import Cluster
    from mrkt.platform.AWS import EC2
    from mrkt.service import DockerViaSSH
    from sys import argv

    services = [
        EC2(DockerViaSSH,
            service_dict={
                "c4.8xlarge": 1,
                "c4.4xlarge": 1,
                "c4.2xlarge": 5,
            },
            sgroup="sg-c86bc4ae",
            keyname="research",
            keyfile="../research.pem",
            clean_action="stop"),
        # DockerViaSSH("0.tcp.ap.ngrok.io",
        # ssh_options=dict(port=13064, username="tefx"),
        # worker_port=12226),
        DockerViaSSH("localhost"),
    ]

    with Cluster(services,
                 image="tefx/mrwsi",
                 image_update=False,
                 image_clean=False) as cluster:
        cluster.sync_dir("./resources/workflows")
        cluster.sync_dir("./test.py")
        cluster.sync_dir("./MrWSI")
        for ccr in argv[1:]:
            wrk_path = "./resources/workflows/random_{}".format(ccr)
            res_path = "./res_ar/{}.res".format(ccr)
            print("solve wrks with ccr={}".format(ccr))
            wrks = list(random_wrks(wrk_path, ""))
            all_results = list(cluster.map(run_alg_on, wrks))
            with open(res_path, "w") as f:
                stat_n_plot(all_results, "hist", "AS", outfile=f)
