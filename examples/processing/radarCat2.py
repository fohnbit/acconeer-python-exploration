from enum import Enum
import numpy as np
from scipy.signal import welch

from acconeer_utils.clients import SocketClient, SPIClient, UARTClient
from acconeer_utils.clients import configs
from acconeer_utils import example_utils
from acconeer_utils.structs import configbase


HALF_WAVELENGTH = 2.445e-3  # m
HISTORY_LENGTH = 2.0  # s
EST_VEL_HISTORY_LENGTH = HISTORY_LENGTH  # s
SD_HISTORY_LENGTH = HISTORY_LENGTH  # s
NUM_SAVED_SEQUENCES = 10
SEQUENCE_TIMEOUT_LENGTH = 0.5  # s
FFT_OVERSAMPLING_FACTOR = 4


def main():
    args = example_utils.ExampleArgumentParser(num_sens=1).parse_args()
    example_utils.config_logging(args)

    if args.socket_addr:
        client = SocketClient(args.socket_addr)
    elif args.spi:
        client = SPIClient()
    else:
        port = args.serial_port or example_utils.autodetect_serial_port()
        client = UARTClient(port)

    sensor_config = get_sensor_config()
    processing_config = get_processing_config()
    sensor_config.sensor = args.sensors

    session_info = client.setup_session(sensor_config)
    print (session_info)

    client.start_streaming()

    interrupt_handler = example_utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")

    processor = Processor(sensor_config, processing_config, session_info)

    while not interrupt_handler.got_signal:
        info, sweep = client.get_next()
        plot_data = processor.process(sweep)

    print("Disconnecting...")
    client.disconnect()


def get_sensor_config():
    config = configs.SparseServiceConfig()

    config.range_interval = [3.2, 4.0]
    config.stepsize = 2
    config.sampling_mode = configs.SparseServiceConfig.SAMPLING_MODE_A
    config.number_of_subsweeps = 256
    config.gain = 0.6
    config.hw_accelerated_average_samples = 60
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
        self.num_subsweeps = sensor_config.number_of_subsweeps
        subsweep_rate = session_info["actual_subsweep_rate"]
        est_update_rate = subsweep_rate / self.num_subsweeps

        self.fft_length = (self.num_subsweeps // 2) * FFT_OVERSAMPLING_FACTOR
        self.num_noise_est_bins = 3
        noise_est_tc = 1.0
        self.min_threshold = 2.5
        self.dynamic_threshold = 0.1

        self.sequence_timeout_count = int(round(SEQUENCE_TIMEOUT_LENGTH * est_update_rate))
        self.noise_est_sf = self.tc_to_sf(noise_est_tc, est_update_rate)
        self.bin_fs = np.fft.rfftfreq(self.fft_length) * subsweep_rate
        self.bin_vs = self.bin_fs * HALF_WAVELENGTH

        num_bins = self.bin_fs.size
        self.noise_est = 0
        self.current_sequence_idle = self.sequence_timeout_count + 1
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
                nperseg=self.num_subsweeps // 2,
                detrend=False,
                axis=0,
                nfft=self.fft_length,
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
            
        if est_vel > 0.2:
            fft = np.fft.rfft(zero_mean_sweep.T * np.hanning(sweep.shape[0]), axis=1)
            abs_fft = np.abs(fft)
            max_depth_index, max_bin = np.unravel_index(abs_fft.argmax(), abs_fft.shape)
            depth = self.depths[max_depth_index]
            print (str(round(est_vel, 1)) + "m/s at " + str(round(depth ,1)) + "m")
            
        self.update_idx += 1


def get_range_depths(sensor_config, session_info):
    range_start = session_info["actual_range_start"]
    range_end = range_start + session_info["actual_range_length"]
    num_depths = session_info["data_length"] // sensor_config.number_of_subsweeps
    return np.linspace(range_start, range_end, num_depths)


if __name__ == "__main__":
    main()
