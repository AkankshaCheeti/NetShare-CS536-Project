import netshare.ray as ray
from netshare import Generator

if __name__ == '__main__':
    ray.config.enabled = True
    ray.init(address="auto")

    generator = Generator(config="pcap/config_example_pcap_nodp.json")
    generator.train_and_generate(work_folder='/nfs/NetShare/results/test')

    ray.shutdown()