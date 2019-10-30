import shlex
import email, smtplib, ssl
from enum import Enum
import numpy as np
from scipy.signal import welch
from threading import Thread
import subprocess
import configparser
import logging
import exifread
import os
from datetime import datetime
import time
from threading import Timer
import signal
import sys

from acconeer_utils.clients import SocketClient, SPIClient, UARTClient
from acconeer_utils.clients import configs
from acconeer_utils import example_utils
from acconeer_utils.structs import configbase

from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.virtual import viewport, sevensegment


HALF_WAVELENGTH = 2.445e-3  # m
FFT_OVERSAMPLING_FACTOR = 4

SETTINGS = configparser.ConfigParser()

# Speedlimit in km/h
SPEEDLIMIT = None
SPEEDLIMIT_TEMP = None
CAMERA = None
CONTEXT = None
LOCKRADAR = None
DIRECTION = ""
IMAGE_FILE_NAME = ""
client = None
EXIT = None
LOCK = None
SENSOR_CONFIG = None
SEG = None

# setup logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

def signal_handler(sig, frame):
    global CAMERA
    global client
    global logging
    
    try:
        logging.info("Disconnect")
        client.disconnect
        CAMERA.exit()
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        
    sys.exit(0)
        
def get_sensor_config():
    global SETTINGS
    config = configs.SparseServiceConfig()
    
    try:
        radar = SETTINGS["Sensor"]
        config.range_interval = [float(radar["range_start"]), float(radar["range_end"])]
        config.stepsize = int(radar["stepsize"])
        config.sampling_mode = configs.SparseServiceConfig.SAMPLING_MODE_A
        config.number_of_subsweeps = int(radar["number_of_subsweeps"])
        config.gain = float(radar["gain"])
        config.hw_accelerated_average_samples = int(radar["hw_accelerated_average_samples"])
        # config.subsweep_rate = 6e3

        # force max frequency
        config.sweep_rate = int(radar["sweep_rate"])
        config.experimental_stitching = False
        
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        
    return config    
    
def main():
    global CAMERA
    global logging
    global client
    global SENSOR_CONFIG
    global SETTINGS
    global SEG
    
    signal.signal(signal.SIGINT, signal_handler)
    try:
        # create seven segment device
        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(serial, cascaded=1)
        SEG = sevensegment(device)
    
        SETTINGS.read("settings.ini")
        
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
        if os.name != 'nt':
            logging.info("set Camera date and time")
            callback_obj = gp.check_result(gp.use_python_logging())
            # open camera connection
            CAMERA = gp.check_result(gp.gp_camera_new())
            gp.check_result(gp.gp_camera_init(CAMERA))
            # get camera details
            abilities = gp.check_result(gp.gp_camera_get_abilities(CAMERA))
            # get configuration tree
            camConfig = gp.check_result(gp.gp_camera_get_config(CAMERA))
            
            # find the date/time setting config item and set it
            if set_datetime(camConfig, abilities.model):
                # apply the changed config
                gp.check_result(gp.gp_camera_set_config(CAMERA, camConfig))
            else:
                logging.error("Could not set date & time")
        
        SENSOR_CONFIG = get_sensor_config()
        SENSOR_CONFIG.sensor = args.sensors
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        
    while True:
        detection()
        time.sleep(3)
    
 

    
def detection():
    global CAMERA
    global logging
    global SPEEDLIMIT_TEMP
    global SETTINGS
    global SPEEDLIMIT    
    global DIRECTION
    global client
    global LOCK
    global EXIT
    global SENSOR_CONFIG
    global CONTINUE
    global SEG
    
    LOCK = False
    CONTINUE = False
    
    try:
        SPEEDLIMIT = float(SETTINGS.get("Speed","Limit"))
        SPEEDLIMIT_TEMP = SPEEDLIMIT
        SEG.text = "MAX " + SPEEDLIMIT    


        logging.info(SENSOR_CONFIG)
        session_info = client.setup_session(SENSOR_CONFIG)
        logging.info(session_info)
        
        client.start_streaming()

        processing_config = get_processing_config()
        
        processor = Processor(SENSOR_CONFIG, processing_config, session_info)


        lastSpeed = np.nan
        lastDistance = 0
        curDirection = "away"
        gotDirection = False
        detection_in_progress = False

        while not LOCK:                    
            info, sweep = client.get_next()        
            plot_data = processor.process(sweep)

            speed = (plot_data["speed"])

            if np.isnan(speed) and np.isnan(lastSpeed):
                continue

            speed = speed * 3.6
            distance = (plot_data["distance"])

            if speed > 0.2 and (lastSpeed != speed or distance != lastDistance):
                SEG.text = SPEEDLIMIT + " " + speed
                if lastDistance != 0 and distance > lastDistance:
                   if not gotDirection:
                        DIRECTION = "away"
                        gotDirection = True
                   curDirection = "away"
                elif lastDistance != 0 and distance < lastDistance:
                    if not gotDirection:
                        DIRECTION = "towards"
                        gotDirection = True
                    curDirection = "towards"
                elif lastDistance != 0 and distance == lastDistance:
                   curDirection = "stay"
                else:
                   curDirection = ""

                logging.info("Speed: " + str(round(speed, 1)) + "km/h in " + str(round(distance, 1)) + "m " + curDirection)
                lastSpeed = speed
                lastDistance = distance
            elif speed < 0.4 and lastSpeed != 0:
                logging.info("No movement")
                lastSpeed = np.nan
                lastDistance = 0

            if speed > SPEEDLIMIT_TEMP:
                SPEEDLIMIT_TEMP = speed
                SEG.text = SPEEDLIMIT + " " + speed
                logging.info("Max Speed: " + str(SPEEDLIMIT_TEMP))
                if not detection_in_progress:
                    detection_in_progress = True
                    
                    threadCaptureImage = Thread(target = captureImage, args=[])
                    threadCaptureImage.start()
                    
                    r = Timer(1.0, lockRadar, (""))
                    r.start()
    except:
        print ("Unexpected error:", sys.exc_info()[0])
    
    try:
        logging.info("Stop streaming")
        client.stop_streaming()
        SEG.text = "Max " + SPEEDLIMIT
    except:
        print ("Unexpected error:", sys.exc_info()[0])
    
    intensity = 2
    while not CONTINUE:
        # logging.info("Waiting...")
        if intensity == 2:
            intensity = 16
        else:
            intensity = 2
            
        seg.device.contrast(intensity * 16)
        time.sleep(0.1)
     
    SEG.text = ""
    seg.device.contrast(16 * 16)

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
        proc = SETTINGS["Proc"]
    
        self.num_subsweeps = sensor_config.number_of_subsweeps
        subsweep_rate = session_info["actual_subsweep_rate"]
        est_update_rate = subsweep_rate / self.num_subsweeps

        self.fft_length = (self.num_subsweeps // 2) * FFT_OVERSAMPLING_FACTOR

        self.num_noise_est_bins = int(proc["num_noise_est_bins"])
        noise_est_tc = float(proc["noise_est_tc"])
        self.min_threshold = float(proc["min_threshold"])
        self.dynamic_threshold = float(proc["dynamic_threshold"])
        
        self.noise_est_sf = self.tc_to_sf(noise_est_tc, est_update_rate)
        self.bin_fs = np.fft.rfftfreq(self.fft_length) * subsweep_rate
        self.bin_vs = self.bin_fs * HALF_WAVELENGTH
        self.update_idx = 0
        self.depths = get_range_depths(sensor_config, session_info)
        self.noise_est = 0
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

        # print speed and distance
        fft = np.fft.rfft(zero_mean_sweep.T * np.hanning(sweep.shape[0]), axis=1)
        abs_fft = np.abs(fft)
        max_depth_index, max_bin = np.unravel_index(abs_fft.argmax(), abs_fft.shape)
        depth = self.depths[max_depth_index]
        
        self.update_idx += 1

        return {
             "speed": est_vel,
             "distance": depth,
        }
      
def get_range_depths(sensor_config, session_info):
    range_start = session_info["actual_range_start"]
    range_end = range_start + session_info["actual_range_length"]
    num_depths = session_info["data_length"] // sensor_config.number_of_subsweeps
    return np.linspace(range_start, range_end, num_depths)
    
def captureImage():
    global CAMERA
    global IMAGE_FILE_NAME
    global logging
    global SETTINGS
    global DIRECTION
    global SPEEDLIMIT_TEMP
    global SPEEDLIMIT
    global CONTINUE
    # try:
    if os.name == 'nt':
        return
    
    # increment Image Counter
    imageCounter = int(SETTINGS["Camera"]["count"])
    imageCounter = imageCounter + 1
    SETTINGS["Camera"]["count"] = str(imageCounter)
    
    # capture the image
    IMAGE_FILE_NAME = 'image' + str(imageCounter)
    # current_time = datetime.now()
    logging.info("Capture Image")
    file_path = CAMERA.capture(gp.GP_CAPTURE_IMAGE)
    logging.info('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    target = os.path.join('.', IMAGE_FILE_NAME + '.jpg')
    logging.info('Copying image to ' + target)
    camera_file = CAMERA.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    
    # incremendet counter saving
    with open('settings.ini', 'w') as configfile:
        SETTINGS.write(configfile)

    # check direction of the movement
    if DIRECTION == "away":
        dir = "A"
    elif DIRECTION == "towards":
        dir = "T"
    else:
        dir = ""
    DIRECTION = ""

    logging.info("Read EXIF data")
    # read EXIF data
    f = open(IMAGE_FILE_NAME + ".jpg", 'rb')
    tags = exifread.process_file(f)
    
    exposure = str(tags["EXIF ExposureTime"])
    iso = str(tags["EXIF ISOSpeedRatings"])
    aperture = "f/" + str(eval(str(tags["EXIF FNumber"])))
    focal = str(tags["EXIF FocalLength"]) + " mm"
    dateTime = str(tags["EXIF DateTimeOriginal"])
    
    logging.info("Start post processing")
    # start post processing
    myCmd = "convert " + IMAGE_FILE_NAME + ".jpg -strokewidth 0 -fill \"rgba( 0, 0, 0, 1 )\" \
    -draw \"rectangle 0,0 6000,300 \" -font helvetica -fill white -pointsize 100 \
    -draw \"text 30,130 'SPEED'\" -fill white -pointsize 100 \
    -draw \"text 30,230 '" + str(round(SPEEDLIMIT_TEMP, 1)) + " km/h'\" -fill white -pointsize 100 \
    -draw \"text 500,130 'DIR'\" -fill white -pointsize 100 \
    -draw \"text 500,230 '" + dir + "'\" -fill white -pointsize 100 \
    -draw \"text 800,130 'DATE'\" -fill white -pointsize 100 \
    -draw \"text 800,230 '" + dateTime + "'\" -fill white -pointsize 100 \
    -draw \"text 1300,130 ' H:M:S'\" -fill white -pointsize 100 \
    -draw \"text 1800,130 'CODE'\" -fill white -pointsize 100 \
    -draw \"text 1800,230 'radarCat'\" -fill white -pointsize 100 \
    -draw \"text 2300,130 'FOTO'\" -fill white -pointsize 100 \
    -draw \"text 2300,230 '" + str(imageCounter) + "'\" -fill white -pointsize 100 \
    -draw \"text 2700,130 'SP.LIMIT'\" -fill white -pointsize 100 \
    -draw \"text 2700,230 '" + str(round(SPEEDLIMIT, 1)) + " km/h'\" -fill white -pointsize 100 \
    -draw \"text 3200,130 'EXPOSURE'\" -fill white -pointsize 100 \
    -draw \"text 3200,230 '" + exposure + "'\" -fill white -pointsize 100 \
    -draw \"text 4100,130 'ISO'\" -fill white -pointsize 100 \
    -draw \"text 4100,230 '" + iso + "'\" -fill white -pointsize 100 \
    -draw \"text 4500,130 'APERTURE'\" -fill white -pointsize 100 \
    -draw \"text 4500,230 '" + aperture + "'\" -fill white -pointsize 100 \
    -draw \"text 5200,130 'FOCAL'\" -fill white -pointsize 100 \
    -draw \"text 5200,230 '" + focal + "'\" \
    radarCat_" + IMAGE_FILE_NAME + ".jpg"
    args = shlex.split(myCmd)
    subprocess.call(args)
  
    # sende image by email
    logging.info("Send Email with Attachment")
    sendEmail(SPEEDLIMIT, IMAGE_FILE_NAME)

    # reset values and restart detection
    SPEEDLIMIT_TEMP = SPEEDLIMIT
    logging.info ("Restart detection")
    
    # copy files and delete after copy
    server = SETTINGS["Server"]
    send_server(server["server"], server["user"], server["password"], server["path"], "radarCat_" + IMAGE_FILE_NAME + ".jpg")
    os.remove("radarCat_" + IMAGE_FILE_NAME + ".jpg")
    os.remove(IMAGE_FILE_NAME + ".jpg")
        
    # except:
    #     print ("Unexpected error:", sys.exc_info()[0])
    
    CONTINUE = True


def lockRadar():
    global LOCK
    global logging
    logging.info("Send LOCK command")
    LOCK = True


def set_datetime(config, model):
    try:
        if model == 'Canon EOS 80D':
            OK, date_config = gp.gp_widget_get_child_by_name(config, 'datetimeutc')
            if OK >= gp.GP_OK:
                now = int(time.time())
                gp.check_result(gp.gp_widget_set_value(date_config, now))
                return True
        OK, sync_config = gp.gp_widget_get_child_by_name(config, 'syncdatetime')
        if OK >= gp.GP_OK:
            gp.check_result(gp.gp_widget_set_value(sync_config, 1))
            return True
        OK, date_config = gp.gp_widget_get_child_by_name(config, 'datetime')
        if OK >= gp.GP_OK:
            widget_type = gp.check_result(gp.gp_widget_get_type(date_config))
            if widget_type == gp.GP_WIDGET_DATE:
                now = int(time.time())
                gp.check_result(gp.gp_widget_set_value(date_config, now))
            else:
                now = time.strftime('%Y-%m-%d %H:%M:%S')
                gp.check_result(gp.gp_widget_set_value(date_config, now))
            return True
    
    except:
        print ("Unexpected error:", sys.exc_info()[0])
    
    return False

def send_server(server, user, password, path, file):
    try:
        logging.info("Copy to local media server")
        subprocess.call(["sshpass","-f",password , "scp", file, user + "@" + server + ":" + path])
    except:
        print ("Unexpected error:", sys.exc_info()[0])

def sendEmail(speedlimit, image_file_name):
    global SETTINGS    
    try:
        email = SETTINGS["Email"]
     
        body = email["body"] + " "  + str(speedlimit) + " km/h"
        message = MIMEMultipart()
        message["From"] = email["sender_email"]
        message["To"] = email["receiver_email"]
        message["Subject"] = email["subject"]

        message.attach(MIMEText(body, "plain"))

        filename = "radarCat_" + image_file_name + ".jpg"
        img_data = open(filename, 'rb').read()

        image = MIMEImage(img_data, name=os.path.basename(filename))
      
        message.attach(image)
        text = message.as_string()

        # Log in to server using secure context and send email#
        s = smtplib.SMTP(email["server"], int(email["port"]))
        s.starttls()
        s.login(email["user"], email["password"])
        s.sendmail(email["sender_email"], email["receiver_email"], text)
        s.quit()    
    except:
        print ("Unexpected error:", sys.exc_info()[0])  
        
if __name__ == "__main__":
    if os.name != 'nt':
        import gphoto2 as gp 
    
    main()
