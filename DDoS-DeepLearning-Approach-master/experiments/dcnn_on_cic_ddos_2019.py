import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.interpolate import make_interp_spline
from data_loaders.Generic_Pacp_Dataset import *
from models.dcnn import *
from trainers.universal_trainer import *

from utils.file_io import list_file
from data_loaders.CIC_DDoS_2019.preprocess_loader import load_label, load_feature, parsing_label
from data_loaders.No_Label_Pcap_Set.preprocess_loader import load_feature as load_feature_without_label
from data_loaders.No_Label_Pcap_Set.preprocess_loader import generate_default_label_dict
from data_loaders.Predict import capture_pcap

mpl.rcParams['toolbar'] = 'None'

logger = logging.getLogger(__name__)

# some variable
SAMPLE_NUMBER = 10000
IS_TRAINING = False
CNN_SHAPE = (100, 160)
CACHE_ROOT = "cache/generic_loader/7-28T05-02/"
PREPROCESSOR_DUMP_PATH = CACHE_ROOT + "combine_set_cache"
# capture config
CAPTURE_FILE = "cache/current_cap.pcap"
INTERFACE = "Intel(R) Wi-Fi 6E AX210 160MHz"
COUNT = 10000
TIMEOUT = 5
CAPTURE_PREPROCESS_DUMP = CACHE_ROOT + "combine_set_cache(predict)"

preprocessor_config = PcapPreprocessorConfig(data_dump_path=PREPROCESSOR_DUMP_PATH,
                                             pkt_in_each_flow_limit=CNN_SHAPE[0],
                                             rewrite=True)

data_loader_config = GenericPcapDataLoaderConfig(preprocessor_dump_path=PREPROCESSOR_DUMP_PATH,
                                                 feature_shape=CNN_SHAPE,
                                                 batch_size=10,
                                                 shuffle_buffer_size=50000,
                                                 return_flow_id=False)

model_config = DCNNModelConfig(feature_size=CNN_SHAPE[1],
                               pkt_each_flow=CNN_SHAPE[0],
                               learning_rate=0.0001,
                               clip_norm=0.1)

trainer_config = UniversalTrainerConfig(epoch=2)

predict_preprocessor_config = PcapPreprocessorConfig(data_dump_path=CAPTURE_PREPROCESS_DUMP,
                                                     pkt_in_each_flow_limit=CNN_SHAPE[0],
                                                     rewrite=True)

predict_data_loader_config = GenericPcapDataLoaderConfig(preprocessor_dump_path=CAPTURE_PREPROCESS_DUMP,
                                                         feature_shape=CNN_SHAPE,
                                                         batch_size=1000,
                                                         shuffle_buffer_size=1,
                                                         return_flow_id=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    if IS_TRAINING:
        pcap_file_directory = "dataset/CIC_DDoS_2019/PCAP/3-11"
        files = list_file(pcap_file_directory)
        files = [pcap_file_directory + "/" + f for f in files]
        files = [x for x in files if 136 >= int(x.split("_")[-1]) >= 107]

        logger.warning("Programme started in train mode!")

        logger.info("Loading normal flow...")
        normal_feature_list = list(load_feature_without_label(["dataset/Normal_Sample/webpage.pcap",
                                                               "dataset/Normal_Sample/webpage_3.pcap",
                                                               "dataset/Normal_Sample/webpage_2.pcap",
                                                               "dataset/Normal_Sample/bilibili_webpage_nextcloudsync.pcap",
                                                               ],
                                                              pkt_in_each_flow_limit=CNN_SHAPE[0],
                                                              sample_limit=SAMPLE_NUMBER))
        normal_label_dict = generate_default_label_dict(normal_feature_list,
                                                        default_label=[1.0, 0.0])
        logger.info("Loading attack flow...")
        attack_label_dict = load_label("dataset/CIC_DDoS_2019/CSV/03-11/UDP.csv", CACHE_ROOT + "label_from_csv_cache")
        attack_feature_list = list(load_feature(files,
                                                pkt_in_each_flow_limit=CNN_SHAPE[0],
                                                label_dict=attack_label_dict,
                                                sample_limit_dict={"BENIGN": 0, "MSSQL": 0, "UDP": SAMPLE_NUMBER}))
        attack_label_dict = parsing_label(attack_label_dict)

        logging.info("Generating dataset...")

        mixed_feature_list = list(attack_feature_list) + list(normal_feature_list)
        mixed_label_dict = attack_label_dict.copy()
        mixed_label_dict.update(normal_label_dict)

        logger.debug("Normal feature list: %s", list(normal_feature_list))
        logger.debug("Attack feature list: %s", list(attack_feature_list))
        logger.debug("Mixed feature list: %s", mixed_feature_list)

        preprocessor = PcapPreprocessor(preprocessor_config, mixed_label_dict, mixed_feature_list)
        data_loader = GenericPcapDataLoader(data_loader_config)
        model = DCNNModel(model_config)
        trainer = UniversalTrainer(model.get_model(), data_loader.get_dataset(), trainer_config)

        for feature, label in data_loader.get_dataset():
            logger.debug("Label: %s", label)

        trainer.train()
        logger.info("Saving weight...")
        trainer.save("logs/CIC_DDoS_2019/release/save.h5")
    else:
        from utils.decision_making import send_attack_ip

        logger.warning("Programme started in predict mode!")
        model = DCNNModel(model_config)
        trainer = UniversalTrainer(model.get_model(), None, trainer_config)
        trainer.load("logs/CIC_DDoS_2019/release/save.h5")

        # capture and predict
        logger.info("Start capture and predict...")
        predict_index = 0

        record_dict = {}

        while True:
            predict_index += 1
            logger.info("Predict turn %s", predict_index)
            logger.info("Capturing...")
            capture_pcap(CAPTURE_FILE, INTERFACE, TIMEOUT, COUNT)
            logger.info("Capture done, generating predict set...")

            predict_feature_list = list(load_feature_without_label([CAPTURE_FILE, ],
                                                                   pkt_in_each_flow_limit=CNN_SHAPE[0],
                                                                   sample_limit=5000))
            predict_label_dict = generate_default_label_dict(predict_feature_list, default_label=[0.0, 1.0])

            logger.debug("Predict feature list: %s", predict_feature_list)
            logger.debug("Predict label dict: %s", predict_label_dict)

            predict_preprocessor = PcapPreprocessor(predict_preprocessor_config, predict_label_dict,
                                                    predict_feature_list)
            predict_set = GenericPcapDataLoader(predict_data_loader_config)

            if predict_set.get_dataset() is not None:
                # try:
                #     trainer.evaluate(predict_set.get_dataset())
                # except TypeError:
                #     logger.error("No data, continue...")

                result_list = []
                for flow_id, flow, label in predict_set.get_dataset():
                    predict_result = np.argmax(trainer.model.predict(flow), axis=-1)
                    result_list.append(np.average(predict_result))
                    logger.debug("Predict flow id: %s, label: %s", flow_id[0].numpy().decode("utf-8"),
                                 predict_result[0])
                    ip_addr = flow_id.numpy()[0].decode("utf-8").split("-")[0]
                    if predict_result[0] == 1:
                        send_attack_ip(ip_addr)

                if result_list:
                    logger.warning("Attack: about %s%%", int(np.average(result_list) * 100))
                    record_dict[time.time()] = int(np.average(result_list) * 100)

                    # draw plot
                    x = np.array([int(x) for x, _ in record_dict.items()])
                    y = np.array([y for _, y in record_dict.items()])
                    x_smooth = np.linspace(np.min(x), np.max(x), 300)
                    try:
                        y_smooth = make_interp_spline(x, y)(x_smooth)
                        y_smooth = np.clip(y_smooth, 0, 100)
                    except ValueError:
                        continue
                    plt.ion()
                    fig = plt.figure("Attack", clear=True, figsize=(1.6 * 5, 0.9 * 5))
                    plt.show()
                    plt.ylim(0, 100)
                    plt.ylabel("Attack Flow (%)")
                    plt.xlabel("Time")
                    plt.plot(x_smooth, y_smooth)
                    plt.grid(False)
                    plt.draw()
                    plt.pause(0.001)

                else:
                    logger.warning("No predict result! The capture file do not contain any useful data.")
