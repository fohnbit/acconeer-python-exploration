import numpy as np

from acconeer_utils.clients import SocketClient, SPIClient, UARTClient
from acconeer_utils.clients import configs
from acconeer_utils import example_utils
from acconeer_utils.structs import configbase


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

    client.start_streaming()

    interrupt_handler = example_utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")

    processor = Processor(sensor_config, processing_config, session_info)

    while not interrupt_handler.got_signal:
        info, sweep = client.get_next()
        speed = processor.process(sweep)
        print (speed)
     break

    print("Disconnecting...")
    client.disconnect()


def get_sensor_config():
    config = configs.SparseServiceConfig()
    config.range_interval = [1.5, 1.58]
    config.sampling_mode = config.SAMPLING_MODE_A
    config.number_of_subsweeps = 64
    config.subsweep_rate = 3e3

    # max frequency
    config.sweep_rate = 100
    config.experimental_stitching = True

    return config


class ProcessingConfiguration(configbase.ProcessingConfig):
    VERSION = 1

    show_data_plot = configbase.BoolParameter(
            label="Show data",
            default_value=True,
            updateable=True,
            order=0,
            )

    show_speed_plot = configbase.BoolParameter(
            label="Show speed on FFT y-axis",
            default_value=False,
            updateable=True,
            order=10,
            )


get_processing_config = ProcessingConfiguration


class Processor:
    def __init__(self, sensor_config, processing_config, session_info):
        # pass
        half_wavelength = 2.445e-3  # m
        subsweep_rate = sensor_config.subsweep_rate
        num_subsweeps = sensor_config.number_of_subsweeps
        self.bin_index_to_speed = half_wavelength * subsweep_rate / num_subsweeps

    def process(self, sweep):
        zero_mean_sweep = sweep - sweep.mean(axis=0, keepdims=True)
        fft = np.fft.rfft(zero_mean_sweep.T * np.hanning(sweep.shape[0]), axis=1)
        abs_fft = np.abs(fft)

        # abs_fft is a matrix with dims abs_fft[depth, freq_bin]
        # we don't care about the depth, so take max over all depths
        asd = np.max(abs_fft, axis=0)

        # asd becomes a vector (asd[freq_bin])
        # now we take the argmax of that
        max_bin = np.argmax(asd)

        speed = max_bin * self.bin_index_to_speed

        return speed

if __name__ == "__main__":
    main()
