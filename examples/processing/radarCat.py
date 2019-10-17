from enum import Enum
import numpy as np
from scipy.signal import welch
from threading import Thread
from time import sleep
import subprocess
import gphoto2 as gp
import logging
import os
from datetime import datetime
import time

from acconeer_utils.clients import SocketClient, SPIClient, UARTClient
from acconeer_utils.clients import configs
from acconeer_utils import example_utils
from acconeer_utils.structs import configbase


HALF_WAVELENGTH = 2.445e-3  # m
NUM_FFT_BINS = 512
HISTORY_LENGTH = 2.0  # s
EST_VEL_HISTORY_LENGTH = HISTORY_LENGTH  # s
SD_HISTORY_LENGTH = HISTORY_LENGTH  # s
NUM_SAVED_SEQUENCES = 10
SEQUENCE_TIMEOUT_COUNT = 10

WAITFORCOMPLETINGSPEEDLIMITDETECTION = None

# Speedlimit in km/h
SPEEDLIMIT = 4
SPEEDLIMIT_TEMP = SPEEDLIMIT
CAMERA = None
CONTEXT = None
# setup logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

def main():
    global CAMERA
    global CONTEXT
    global logging
    
    args = example_utils.ExampleArgumentParser(num_sens=1).parse_args()
    example_utils.config_logging(args)
    logging.info("radarCat starting with args " + str(args))

    if args.socket_addr:
        client = SocketClient(args.socket_addr)
    elif args.spi:
        client = SPIClient()
    else:
        port = args.serial_port or example_utils.autodetect_serial_port()
        client = UARTClient(port)

    # setup Camera date and time
    logging.info("set Camera date and time")
    subprocess.call(["gphoto2","--set-config", "datetime=now"])
    
    gp.check_result(gp.use_python_logging())
    CONTEXT = gp.gp_context_new()
    CAMERA = gp.check_result(gp.gp_camera_new())
    gp.check_result(gp.gp_camera_init(CAMERA, CONTEXT))

    sensor_config = get_sensor_config()
    processing_config = get_processing_config()
    sensor_config.sensor = args.sensors

    session_info = client.setup_session(sensor_config)
    logging.info(session_info)

    client.start_streaming()

    interrupt_handler = example_utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")

    processor = Processor(sensor_config, processing_config, session_info)

    global SPEEDLIMIT_TEMP
    global WAITFORCOMPLETINGSPEEDLIMITDETECTION
    lastSpeed = 0
    
    while not interrupt_handler.got_signal:
        info, sweep = client.get_next()
        plot_data = processor.process(sweep)
        
        speed = (plot_data["speed"]) * 3.6
        distance = (plot_data["distance"])
 
        if speed > 1 and lastSpeed != speed:
            logging.info("Speed: " + str(round(speed, 1)) + "km/h in " + str(round(distance, 1)) + "m")
            lastSpeed = speed
        
        if speed > SPEEDLIMIT_TEMP:
            SPEEDLIMIT_TEMP = speed
            logging.info("Maximal current Speed: " + str(SPEEDLIMIT_TEMP))
            if not WAITFORCOMPLETINGSPEEDLIMITDETECTION:
                WAITFORCOMPLETINGSPEEDLIMITDETECTION = True
                
                threadCaptureImage = Thread(target = captureImage, args=[])
                threadCaptureImage.start()
    
                threadSendRadarCatImage = Thread(target = sendRadarCatImage, args=[])
                threadSendRadarCatImage.start()
                
    print("Disconnecting...")
    client.disconnect()


def get_sensor_config():
    config = configs.SparseServiceConfig()

    config.range_interval = [3.00, 3.20]
    config.stepsize = 3
    config.sampling_mode = configs.SparseServiceConfig.SAMPLING_MODE_A
    config.number_of_subsweeps = NUM_FFT_BINS
    config.gain = 0.5
    config.hw_accelerated_average_samples = 60
    # config.subsweep_rate = 6e3

    # force max frequency
    config.sweep_rate = 200
    config.experimental_stitching = True

    return config


class ProcessingConfiguration(configbase.ProcessingConfig):
    VERSION = 1

    class SpeedUnit(Enum):
        METER_PER_SECOND = ("m/s", 1)
        KILOMETERS_PER_HOUR = ("km/h", 3.6)
        MILES_PER_HOUR = ("mph", 2.237)

        @property
        def label(self):
            return self.value[0]

        @property
        def scale(self):
            return self.value[1]

    min_speed = configbase.FloatParameter(
            label="Minimum speed",
            unit="m/s",
            default_value=0.2,
            limits=(0, 5),
            decimals=1,
            updateable=True,
            order=0,
            )

    shown_speed_unit = configbase.EnumParameter(
            label="Speed unit",
            default_value=SpeedUnit.METER_PER_SECOND,
            enum=SpeedUnit,
            updateable=True,
            order=100,
            )

    show_data_plot = configbase.BoolParameter(
            label="Show data",
            default_value=False,
            updateable=True,
            order=110,
            )

    show_sd_plot = configbase.BoolParameter(
            label="Show spectral density",
            default_value=True,
            updateable=True,
            order=120,
            )

    show_vel_history_plot = configbase.BoolParameter(
            label="Show speed history",
            default_value=True,
            updateable=True,
            order=130,
            )


get_processing_config = ProcessingConfiguration


class Processor:
    def __init__(self, sensor_config, processing_config, session_info):
        subsweep_rate = session_info["actual_subsweep_rate"]
        est_update_rate = subsweep_rate / sensor_config.number_of_subsweeps

        self.nperseg = NUM_FFT_BINS // 2
        self.num_noise_est_bins = 3
        noise_est_tc = 1.0
        self.min_threshold = 4.0
        self.dynamic_threshold = 0.1

        est_vel_history_size = int(round(est_update_rate * EST_VEL_HISTORY_LENGTH))
        sd_history_size = int(round(est_update_rate * SD_HISTORY_LENGTH))
        num_bins = NUM_FFT_BINS // 2 + 1
        self.noise_est_sf = self.tc_to_sf(noise_est_tc, est_update_rate)
        self.bin_fs = np.fft.rfftfreq(NUM_FFT_BINS) * subsweep_rate
        self.bin_vs = self.bin_fs * HALF_WAVELENGTH

        self.nasd_history = np.zeros([sd_history_size, num_bins])
        self.est_vel_history = np.full(est_vel_history_size, np.nan)
        self.belongs_to_last_sequence = np.zeros(est_vel_history_size, dtype=bool)
        self.noise_est = 0
        self.current_sequence_idle = SEQUENCE_TIMEOUT_COUNT + 1
        self.sequence_vels = np.zeros(NUM_SAVED_SEQUENCES)
        self.update_idx = 0

        self.depths = get_range_depths(sensor_config, session_info)
        
        self.update_processing_config(processing_config)

    def update_processing_config(self, processing_config):
        self.min_speed = processing_config.min_speed

    def tc_to_sf(self, tc, fs):
        if tc <= 0.0:
            return 0.0

        return np.exp(-1.0 / (tc * fs))

    def dynamic_sf(self, static_sf):
        return min(static_sf, 1.0 - 1.0 / (1.0 + self.update_idx))

    def process(self, sweep):
        # Basic speed estimate

        zero_mean_sweep = sweep - sweep.mean(axis=0, keepdims=True)

        _, psds = welch(
                zero_mean_sweep,
                nperseg=self.nperseg,
                detrend=False,
                axis=0,
                nfft=NUM_FFT_BINS,
                )

        psd = np.max(psds, axis=1)
        asd = np.sqrt(psd)

        inst_noise_est = np.mean(asd[(-self.num_noise_est_bins - 1):-1])
        sf = self.dynamic_sf(self.noise_est_sf)
        self.noise_est = sf * self.noise_est + (1.0 - sf) * inst_noise_est

        nasd = asd / self.noise_est

        threshold = max(self.min_threshold, np.max(nasd) * self.dynamic_threshold)
        over = nasd > threshold
        est_idx = np.where(over)[0][-1] if np.any(over) else np.nan

        if est_idx > 0:  # evaluates to false if nan
            est_vel = self.bin_vs[est_idx]
        else:
            est_vel = np.nan

        if est_vel < self.min_speed:  # evaluates to false if nan
            est_vel = np.nan

        # print speed and distance
        fft = np.fft.rfft(zero_mean_sweep.T * np.hanning(sweep.shape[0]), axis=1)
        abs_fft = np.abs(fft)
        max_depth_index, max_bin = np.unravel_index(abs_fft.argmax(), abs_fft.shape)
        depth = self.depths[max_depth_index]
       
        # print ("Speed: " + str(est_vel) + " m/s, Distance: " + str(depth))
        
        # Sequence

        self.belongs_to_last_sequence = np.roll(self.belongs_to_last_sequence, -1)

        if np.isnan(est_vel):
            self.current_sequence_idle += 1
        else:
            if self.current_sequence_idle > SEQUENCE_TIMEOUT_COUNT:
                self.sequence_vels = np.roll(self.sequence_vels, -1)
                self.sequence_vels[-1] = est_vel
                self.belongs_to_last_sequence[:] = False

            self.current_sequence_idle = 0
            self.belongs_to_last_sequence[-1] = True

            if est_vel > self.sequence_vels[-1]:
                self.sequence_vels[-1] = est_vel

        # Data for plots

        self.est_vel_history = np.roll(self.est_vel_history, -1, axis=0)
        self.est_vel_history[-1] = est_vel

        if np.all(np.isnan(self.est_vel_history)):
            output_vel = 0
        else:
            output_vel = np.nanmax(self.est_vel_history)

        self.nasd_history = np.roll(self.nasd_history, -1, axis=0)
        self.nasd_history[-1] = nasd

        nasd_temporal_max = np.max(self.nasd_history, axis=0)

        temporal_max_threshold = max(
            self.min_threshold, np.max(nasd_temporal_max) * self.dynamic_threshold)

        self.update_idx += 1
      
        return {
            "sweep": sweep,
            "sd": nasd_temporal_max,
            "sd_threshold": temporal_max_threshold,
            "vel_history": self.est_vel_history,
            "vel": output_vel,
            "speed": est_vel,
            "distance": depth,
            "sequence_vels": self.sequence_vels,
            "belongs_to_last_sequence": self.belongs_to_last_sequence,
        }

def get_range_depths(sensor_config, session_info):
    range_start = session_info["actual_range_start"]
    range_end = range_start + session_info["actual_range_length"]
    num_depths = session_info["data_length"] // sensor_config.number_of_subsweeps
    return np.linspace(range_start, range_end, num_depths)
    
def captureImage():
    global CAMERA
    global CONTEXT
    global logging
    
    current_time = datetime.now()
    logging.info("Capture Image")
    file_path = gp.check_result(gp.gp_camera_capture(
        CAMERA, gp.GP_CAPTURE_IMAGE, CONTEXT))
    logging.info('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    target = os.path.join('.', file_path.name)
    print ('Copying image to', target)
    camera_file = gp.check_result(gp.gp_camera_file_get(
            CAMERA, file_path.folder, file_path.name,
            gp.GP_FILE_TYPE_NORMAL, CONTEXT))
    gp.check_result(gp.gp_file_save(camera_file, target))
    # subprocess.call(['xdg-open', target])
    gp.check_result(gp.gp_camera_exit(CAMERA, CONTEXT))

    logging.info("Write capture date/time to file")
    f = open("captureDateTime.txt", "w")
    f.write(str(current_time))
    f.close()

def sendRadarCatImage(): 
    logging.info ("Lock radar until image is sendet")
    sleep(10)
    global WAITFORCOMPLETINGSPEEDLIMITDETECTION
    global SPEEDLIMIT_TEMP
    global SPEEDLIMIT   


    logging.info("Write max Speed to file: " + str(SPEEDLIMIT_TEMP))
    f = open("speed.txt", "w")
    f.write(str(round(SPEEDLIMIT_TEMP, 1)) + " km/h")
    f.close()
    
    logging.info("Write Speedlimit to file: " + str(SPEEDLIMIT))
    f = open("speedLimit.txt", "w")
    f.write(str(round(SPEEDLIMIT, 1)) + " km/h")
    f.close()
    
    logging.info("Start Postprocessing")
    myCmd = './postProcessing.sh'
    subprocess.call([myCmd])
    
    logging.info("Send Email with Attachment")
    myCmd = './sendmail.sh'
    subprocess.call([myCmd])

    SPEEDLIMIT_TEMP = SPEEDLIMIT
    WAITFORCOMPLETINGSPEEDLIMITDETECTION = None

    logging.info ("Release radar lock")
    
if __name__ == "__main__":
    main()
