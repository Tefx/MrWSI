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

    if len(argv) > 1:
        wrk_path = argv[1]
    else:
        wrk_path = "./resources/workflows/random"
    wrks = list(random_wrks(wrk_path, ""))
    with Cluster(services,
                 image="tefx/mrwsi",
                 image_update=False,
                 image_clean=False) as cluster:
        cluster.sync_dir(wrk_path)
        cluster.sync_dir("./test.py")
        cluster.sync_dir("./MrWSI")
        all_results = list(cluster.map(run_alg_on, wrks))

    stat_n_plot(all_results, "hist", "AS")
