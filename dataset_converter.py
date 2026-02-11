from sre_parse import BRANCH
from loguru import logger
import os
import sys
from pathlib import Path
import json
from channels_definition import *
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from pyquaternion import Quaternion as pyQuaternion
from scipy.interpolate import interp1d
import shutil
import tarfile
import time
from merge_lerobot_dataset import merge_datasets
import multiprocessing
import cv2
from cv_bridge import CvBridge
# default: ROS 2
USE_ROS1 = bool(int(os.getenv('USE_ROS1', 0)))
# default: mp4 video
SAVE_VIDEO = bool(int(os.getenv('SAVE_VIDEO', 1)))
# default: AV1 codec
USE_H264 = bool(int(os.getenv('USE_H264', 0)))
# default: original file
USE_COMPRESSION = bool(int(os.getenv('USE_COMPRESSION', 0)))
# default: compute image stats
IS_COMPUTE_EPISODE_STATS_IMAGE = bool(int(os.getenv('IS_COMPUTE_EPISODE_STATS_IMAGE', 1)))
# default: 4 processes
MAX_PROCESSES = int(os.getenv('MAX_PROCESSES', 6))
# default: original description (possibly Chinese)
USE_TRANSLATION = bool(int(os.getenv('USE_TRANSLATION', 0)))

logger.info("env variables")
logger.info(f"USE_ROS1: {USE_ROS1}")
logger.info(f"SAVE_VIDEO: {SAVE_VIDEO}")
logger.info(f"USE_H264: {USE_H264}")
logger.info(f"USE_COMPRESSION: {USE_COMPRESSION}")
logger.info(f"IS_COMPUTE_EPISODE_STATS_IMAGE: {IS_COMPUTE_EPISODE_STATS_IMAGE}")
logger.info(f"MAX_PROCESSES: {MAX_PROCESSES}")
logger.info(f"USE_TRANSLATION: {USE_TRANSLATION}")

# Configure Loguru to avoid duplicate logs
# Remove the default sink before adding our own, otherwise messages
# may appear twice (default stderr sink + added stderr sink).
try:
    logger.remove()
    logger.add(sys.stderr, enqueue=True)
except Exception:
    pass

import os
import time
import hashlib
import requests
import json
from pathlib import Path
import subprocess
import random


def cal_auth(ak, sk):
    timestamp_seconds = time.time()
    timestamp_milliseconds = int(timestamp_seconds * 1e3)
    string_to_encrypt = sk + "," + str(timestamp_milliseconds)
    encoded_string = string_to_encrypt.encode('utf-8')
    sha256_hash = hashlib.sha256()
    sha256_hash.update(encoded_string)
    encrypted_string = sha256_hash.hexdigest()
    header = "Digest " + ak + ";" + str(timestamp_milliseconds) + ";" + encrypted_string
    
    return header

def request_with_retry(method, url, headers=None, data=None, json_body=None, timeout=120,
    max_retries=5, backoff_base_seconds=0.5, max_backoff_seconds=8.0,
    retry_on_status=(429, 500, 502, 503, 504)):
    """
    使用指数退避+随机抖动的重试机制封装 requests.request。

    参数:
        method: HTTP 方法，如 "GET"、"POST"。
        url: 请求地址。
        headers: 请求头。
        data: 表单或字节数据。
        json_body: JSON 负载（与 data 互斥）。
        timeout: 每次请求的超时时间（秒）。
        max_retries: 最大重试次数（不含首次）。
        backoff_base_seconds: 退避基数，实际退避为 base * 2^attempt，并加入抖动。
        max_backoff_seconds: 退避最大值上限。
        retry_on_status: 需要重试的 HTTP 状态码集合。

    返回:
        requests.Response 对象。
    """
    attempt_index = 0
    while True:
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                data=data,
                json=json_body,
                timeout=timeout,
            )
            # 对特定 HTTP 状态码进行重试
            if response.status_code in retry_on_status and attempt_index < max_retries:
                # 计算指数退避时间，加入[0, backoff]的抖动
                backoff_no_jitter = min(
                    max_backoff_seconds,
                    backoff_base_seconds * (2 ** attempt_index)
                )
                sleep_seconds = random.uniform(0, backoff_no_jitter)
                print(f"request retryable status {response.status_code}, retrying in {sleep_seconds:.2f}s (attempt {attempt_index + 1}/{max_retries})")
                time.sleep(sleep_seconds)
                attempt_index += 1
                continue
            # 非重试状态码或已达到重试上限
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt_index >= max_retries:
                # 达到重试上限，抛出异常
                raise
            backoff_no_jitter = min(
                max_backoff_seconds,
                backoff_base_seconds * (2 ** attempt_index)
            )
            sleep_seconds = random.uniform(0, backoff_no_jitter)
            print(f"request exception: {exc}, retrying in {sleep_seconds:.2f}s (attempt {attempt_index + 1}/{max_retries})")
            time.sleep(sleep_seconds)
            attempt_index += 1

def get_raw_data_meta():
    ak = os.getenv('EDP_AK')
    sk = os.getenv('EDP_SK')
    dataset_name = os.getenv('RAW_DATA_SET_NAME')
    version = os.getenv('VERSION')
    cache_dir = "/edp-workspace/instance-env/"
    url = f"https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/training-data-set/get-meta?rawDataSetName={dataset_name}&version={version}"
    payload = {}
    cal_auth_value = cal_auth(ak, sk)
    headers = {
        'accept': '*/*',
        'Authorization': cal_auth_value
    }
    response = request_with_retry("GET", url, headers=headers, data=payload)
    raw_data_meta_json = json.loads(response.text)
    output_dir = os.getenv('TRAINING_DATA_SET_DIR')
    return (raw_data_meta_json, cache_dir, output_dir)

def get_raw_data_by_bag_name(bag_name):
    ak = os.getenv('EDP_AK')
    sk = os.getenv('EDP_SK')
    url = "https://edp.galaxea-ai.com/edp-app-be/backend/v1/business/raw-data/query"
    payload = json.dumps({
        "bagName": bag_name,
        "pageNum": 1,
        "pageSize": -1
    })
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': cal_auth(ak, sk)
    }
    response = request_with_retry("POST", url, headers=headers, data=payload)
    data = json.loads(response.text)
    return data
# get_raw_data_by_bag_name("RB250417001_20250805214344892_RAW.mcap")

    

class DataConverter:
    def __init__(
            self, 
            robot_type, 
            sample_mcap_path, 
            dataset_name, 
            output_dir, 
            use_ros1 = False, 
            save_video = False, 
            use_h264 = False, 
            use_compression = False, 
            is_compute_episode_stats_image = True,
            max_processes = 4,
            use_translation = False,
        ):

        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
        self.robot_type = robot_type
        logger.info(f'Robot type is: {self.robot_type}')
        self.shape_of_images = dict()
        
        self.use_ros1 = use_ros1
        self.save_video = save_video
        self.use_h264 = use_h264
        self.use_compression = use_compression
        self.is_compute_episode_stats_image = is_compute_episode_stats_image
        self.max_processes = max_processes
        self.use_translation = use_translation

        self.fps_dict = {}

        if not self.use_ros1:
            # check if wrist camera topic has "_rect", can be cleaned by ATC 2.1.5
            reader = SequentialReader()
            storage_options = StorageOptions(uri=sample_mcap_path, storage_id="mcap")
            converter_options = ConverterOptions()
            reader.open(storage_options, converter_options)
            all_topics = [topic.name for topic in reader.get_all_topics_and_types()]
            self.RGB_WRIST_LEFT_TOPIC = RGB_WRIST_LEFT_TOPIC
            self.RGB_WRIST_RIGHT_TOPIC = RGB_WRIST_RIGHT_TOPIC
            if self.RGB_WRIST_LEFT_TOPIC not in all_topics or self.RGB_WRIST_RIGHT_TOPIC not in all_topics:
                self.RGB_WRIST_LEFT_TOPIC = self.RGB_WRIST_LEFT_TOPIC.replace("image_raw", "image_rect_raw")
                self.RGB_WRIST_RIGHT_TOPIC = self.RGB_WRIST_RIGHT_TOPIC.replace("image_raw", "image_rect_raw")
            assert self.RGB_WRIST_LEFT_TOPIC in all_topics and self.RGB_WRIST_RIGHT_TOPIC in all_topics
        else:
            self.RGB_WRIST_LEFT_TOPIC = RGB_WRIST_LEFT_TOPIC
            self.RGB_WRIST_RIGHT_TOPIC = RGB_WRIST_RIGHT_TOPIC

        self.RGB_TOPICS = [
            self.RGB_WRIST_LEFT_TOPIC, 
            self.RGB_WRIST_RIGHT_TOPIC, 
            RGB_HEAD_RIGHT_TOPIC
        ]
        self.DEPTH_TOPICS = [
            DEPTH_HEAD_TOPIC, 
            DEPTH_LEFT_TOPIC, 
            DEPTH_RIGHT_TOPIC
        ]
        self.JOINT_TOPICS = [
            JOINT_OBS_LEFT_TOPIC, 
            JOINT_OBS_RIGHT_TOPIC, 
            GRIPPER_OBS_LEFT_TOPIC, 
            GRIPPER_OBS_RIGHT_TOPIC, 
            CHASSIS_OBS_TOPIC, 
            TORSO_OBS_TOPIC, 
            JOINT_ACTION_LEFT_TOPIC, 
            JOINT_ACTION_RIGHT_TOPIC, 
            TORSO_ACTION_TOPIC
        ]
        self.POSE_TOPICS = [
            EE_POSE_OBS_LEFT_TOPIC, 
            EE_POSE_OBS_RIGHT_TOPIC, 
        ]
        if self.robot_type == "r1pro":
            self.POSE_TOPICS.append(EE_POSE_ACTION_LEFT_TOPIC)
            self.POSE_TOPICS.append(EE_POSE_ACTION_RIGHT_TOPIC)
        self.GRIPPER_TOPICS = [
            GRIPPER_ACTION_LEFT_TOPIC, 
            GRIPPER_ACTION_RIGHT_TOPIC
        ]
        self.TWIST_TOPICS = [CHASSIS_ACTION_TOPIC]
        if self.robot_type == "r1lite":
            self.TWIST_TOPICS.append(TORSO_ACTION_SPEED_TOPIC)
        self.CONTROL_TOPICS = [
            JOINT_CONTROL_ACTION_LEFT_TOPIC, 
            JOINT_CONTROL_ACTION_RIGHT_TOPIC, 
            GRIPPER_CONTROL_ACTION_LEFT_TOPIC, 
            GRIPPER_CONTROL_ACTION_RIGHT_TOPIC, 
            CHASSIS_CONTROL_ACTION_TOPIC, 
            TORSO_CONTROL_ACTION_TOPIC
        ]
        self.IMU_TOPICS = [CHASSIS_IMU_TOPIC]
        self.TARGET_TOPICS = {
            GRIPPER_ACTION_LEFT_TOPIC: [],
            GRIPPER_ACTION_RIGHT_TOPIC: [],
            EE_POSE_OBS_LEFT_TOPIC: [],
            EE_POSE_OBS_RIGHT_TOPIC: [],
            JOINT_OBS_LEFT_TOPIC: [],
            JOINT_OBS_RIGHT_TOPIC: [],
            JOINT_ACTION_LEFT_TOPIC: [],
            JOINT_ACTION_RIGHT_TOPIC: [],
            GRIPPER_OBS_LEFT_TOPIC: [],
            GRIPPER_OBS_RIGHT_TOPIC: [],
            RGB_HEAD_LEFT_TOPIC: [],
            RGB_HEAD_RIGHT_TOPIC: [],
            self.RGB_WRIST_LEFT_TOPIC: [],
            self.RGB_WRIST_RIGHT_TOPIC: [],
            DEPTH_HEAD_TOPIC: [],
            DEPTH_LEFT_TOPIC: [],
            DEPTH_RIGHT_TOPIC: [],
            CHASSIS_ACTION_TOPIC: [],
            TORSO_ACTION_TOPIC: [],
            CHASSIS_OBS_TOPIC: [],
            CHASSIS_IMU_TOPIC: [],
            TORSO_OBS_TOPIC: [],
            JOINT_CONTROL_ACTION_LEFT_TOPIC: [],
            JOINT_CONTROL_ACTION_RIGHT_TOPIC: [],
            GRIPPER_CONTROL_ACTION_LEFT_TOPIC: [],
            GRIPPER_CONTROL_ACTION_RIGHT_TOPIC: [],
            CHASSIS_CONTROL_ACTION_TOPIC: [],
            TORSO_CONTROL_ACTION_TOPIC: []
        }
        if self.robot_type == "r1pro":
            self.TARGET_TOPICS[EE_POSE_ACTION_LEFT_TOPIC] = []
            self.TARGET_TOPICS[EE_POSE_ACTION_RIGHT_TOPIC] = []
        if self.robot_type == "r1lite":
            self.TARGET_TOPICS[TORSO_ACTION_SPEED_TOPIC] = []
        if self.robot_type == "r1pro":
            self.arm_dof = 7
        else:
            self.arm_dof = 6

    def extract(self, bag_file):
        if not self.use_ros1:
            return self.extract_ros2(bag_file)
        else:
            return self.extract_ros1(bag_file)

    def extract_ros1(self, bag_file):
        time_start = time.time()
        extracted_msgs = {topic : [] for topic in self.TARGET_TOPICS}
        bag = rosbag.Bag(bag_file)
        for topic, msg, t in bag.read_messages():
            if topic in extracted_msgs.keys():
                extracted_msgs[topic].append(msg)
        bag.close()
        time_end = time.time()
        logger.info(f"extract_ros1 time: {time_end - time_start} seconds")
        return extracted_msgs

    def extract_ros2(self, mcap_file):
        time_start = time.time()
        mcap_name = os.path.basename(mcap_file)
        # logger.info(f"Loading {mcap_name} mcap file.")
        extracted_msgs = {topic : [] for topic in self.TARGET_TOPICS}
        reader = SequentialReader()
        storage_options = StorageOptions(uri=mcap_file, storage_id="mcap")
        converter_options = ConverterOptions()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            msg_type = type_map.get(topic)
            
            if not msg_type:
                logger.warning(f'Unknown topic type: {topic}')
                continue

            try:
                if topic in extracted_msgs.keys():
                    module_name, class_name = msg_type.rsplit('/', 1)
                    module = __import__(f'{module_name.replace("/", ".")}', fromlist=[class_name])
                    msg_class = getattr(module, class_name)
                    msg = deserialize_message(data, msg_class)
                    # msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                    extracted_msgs[topic].append(msg)
            except Exception as e:
                logger.error(f'Error processing {topic}: {str(e)}')
        time_end = time.time()
        logger.info(f"extract_ros2 time: {time_end - time_start} seconds")
        return extracted_msgs
    
    def msg_to_timestamp(self, msg):
        if not self.use_ros1:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        else:
            timestamp = msg.header.stamp.to_sec()
        return timestamp

    def create_features(self, frame_sample):
        features = {}
        # RGB
        image_dtype = "video" if self.save_video else "image"
        features["observation.images.head_rgb"] = {
            "dtype": image_dtype,
            "shape": self.shape_of_images["HEAD_LEFT_RGB"],
            "names": ["height", "width", "channels"],
        }

        features["observation.images.head_right_rgb"] = {
            "dtype": image_dtype,
            "shape": self.shape_of_images["HEAD_LEFT_RGB"],
            "names": ["height", "width", "channels"],
        }

        features["observation.images.left_wrist_rgb"] = {
            "dtype": image_dtype,
            "shape": self.shape_of_images["WRIST_LEFT_RGB"],
            "names": ["height", "width", "channels"],
        }

        features["observation.images.right_wrist_rgb"] = {
            "dtype": image_dtype,
            "shape": self.shape_of_images["WRIST_RIGHT_RGB"],
            "names": ["height", "width", "channels"],
        }

        # Arm Joints
        arm_feat = {
            "dtype": "float64",
            "shape": (self.arm_dof,),
            "names": None
        }
        features["observation.state.left_arm"] = arm_feat.copy()
        features["observation.state.left_arm"]["names"] = [JOINT_OBS_LEFT_TOPIC+f".position[{i}]" for i in range(self.arm_dof)]
        features["observation.state.left_arm.velocities"] = arm_feat.copy()
        features["observation.state.left_arm.velocities"]["names"] = [JOINT_OBS_LEFT_TOPIC+f".velocity[{i}]" for i in range(self.arm_dof)]
        features["observation.state.right_arm"] = arm_feat.copy()
        features["observation.state.right_arm"]["names"] = [JOINT_OBS_RIGHT_TOPIC+f".position[{i}]" for i in range(self.arm_dof)]
        features["observation.state.right_arm.velocities"] = arm_feat.copy()
        features["observation.state.right_arm.velocities"]["names"] = [JOINT_OBS_RIGHT_TOPIC+f".velocity[{i}]" for i in range(self.arm_dof)]

        imu_names = [
            ".orientation.x",
            ".orientation.y",
            ".orientation.z",
            ".orientation.w",
            ".angular_velocity.x",
            ".angular_velocity.y",
            ".angular_velocity.z",
            ".linear_acceleration.x",
            ".linear_acceleration.y",
            ".linear_acceleration.z"
        ]
        # Chassis
        features["observation.state.chassis.imu"] = {
            "dtype": "float64",
            "shape": (10,),
            "names": [CHASSIS_IMU_TOPIC+name for name in imu_names]
        }

        chassis_obs_names = [
            ".position[0]",
            ".position[1]",
            ".position[2]",
            ".velocity[0]",
            ".velocity[1]",
            ".velocity[2]",
        ]
        features["observation.state.chassis"] = {
            "dtype": "float64",
            "shape": (3,),
            "names": [CHASSIS_OBS_TOPIC+name for name in chassis_obs_names[:3]]
        }
        features["observation.state.chassis.velocities"] = {
            "dtype": "float64",
            "shape": (3,),
            "names": [CHASSIS_OBS_TOPIC+name for name in chassis_obs_names[3:]]
        }

        # Torso
        torso_obs_names = [
            ".position[0]",
            ".position[1]",
            ".position[2]",
            ".position[3]",
            ".velocity[0]",
            ".velocity[1]",
            ".velocity[2]",
            ".velocity[3]",
        ]
        features["observation.state.torso"] = {
            "dtype": "float64",
            "shape": (4,),
            "names": [TORSO_OBS_TOPIC+name for name in torso_obs_names[:4]]
        }

        features["observation.state.torso.velocities"] = {
            "dtype": "float64",
            "shape": (4,),
            "names": [TORSO_OBS_TOPIC+name for name in torso_obs_names[4:]]
        }

        # Gripper
        features["observation.state.left_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_OBS_LEFT_TOPIC+".position[0]"]
        }

        features["observation.state.right_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_OBS_RIGHT_TOPIC+".position[0]"]
        }

        # EE
        eef_pose = {
            "dtype": "float64",
            "shape": (7,),
            "names": None
        }
        pose_names = [
            ".pose.position.x",
            ".pose.position.y",
            ".pose.position.z",
            ".pose.orientation.x",
            ".pose.orientation.y",
            ".pose.orientation.z",
            ".pose.orientation.w"
        ]
        features["observation.state.left_ee_pose"] = eef_pose.copy()
        features["observation.state.left_ee_pose"]["names"] = [EE_POSE_OBS_LEFT_TOPIC+name for name in pose_names]
        features["observation.state.right_ee_pose"] = eef_pose.copy()
        features["observation.state.right_ee_pose"]["names"] = [EE_POSE_OBS_RIGHT_TOPIC+name for name in pose_names]

        # Actions
        if self.robot_type == "r1pro":
            features["action.left_ee_pose"] = eef_pose.copy()
            features["action.left_ee_pose"]["names"] = [EE_POSE_ACTION_LEFT_TOPIC+name for name in pose_names]
            features["action.right_ee_pose"] = eef_pose.copy()
            features["action.right_ee_pose"]["names"] = [EE_POSE_ACTION_RIGHT_TOPIC+name for name in pose_names]
        
        features["action.left_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_ACTION_LEFT_TOPIC+".position[0]"]
        }
        features["action.right_gripper"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": [GRIPPER_ACTION_RIGHT_TOPIC+".position[0]"]
        }

        chassis_twist_names = [
            ".twist.linear.x",
            ".twist.linear.y",
            ".twist.linear.z",
            ".twist.angular.x",
            ".twist.angular.y",
            ".twist.angular.z",
        ]
        features["action.chassis.velocities"] = {
            "dtype": "float64", 
            "shape": (6,), 
            "names": [CHASSIS_ACTION_TOPIC+name for name in chassis_twist_names]
        }

        # NOTE: torso will have two different control types, and
        # will not record both of them in the same episode.
        if "action.torso" in frame_sample:
            features["action.torso"] = {
                "dtype": "float64",
                "shape": (4,), 
                "names": [TORSO_ACTION_TOPIC+f".position[{i}]" for i in range(4)]
            }

        if "action.torso.velocities" in frame_sample:
            pose_names = [
                ".twist.linear.x",
                ".twist.linear.y",
                ".twist.linear.z",
                ".twist.angular.x",
                ".twist.angular.y",
                ".twist.angular.z",
            ]
            features["action.torso.velocities"] = {
                "dtype": "float64",
                "shape": (6,),
                "names": [TORSO_ACTION_SPEED_TOPIC+name for name in pose_names]
            }
        
        features["action.left_arm"] = {
                "dtype": "float64",
                "shape": (self.arm_dof,),
                "names": None
            }
        features["action.left_arm"]["names"] = [JOINT_ACTION_LEFT_TOPIC+f".position[{i}]" for i in range(self.arm_dof)]
        features["action.right_arm"] = {
                "dtype": "float64",
                "shape": (self.arm_dof,),
                "names": None
            }
        features["action.right_arm"]["names"] = [JOINT_ACTION_RIGHT_TOPIC+f".position[{i}]" for i in range(self.arm_dof)]
        
        return features

    def process_all(self, mcaps_dict: dict):
        start_time = time.time()
        mcaps_dict = mcaps_dict["rawDataList"]
        total_files = len(mcaps_dict)
        logger.info("processing num: ", self.max_processes)
        args_list = [(idx, mcap_file) for idx, mcap_file in enumerate(mcaps_dict)]
        # self.process(0, mcaps_dict[0])
        if self.max_processes <= 1:
            for args in args_list:
                self.process_wrapper(*args)
        else:
            with multiprocessing.Pool(processes=self.max_processes, maxtasksperchild=1) as pool:
                pool.starmap(self.process_wrapper, args_list)
        
        process_time = time.time() - start_time
        
        start_time = time.time()
        self.merge_subdataset()
        merge_time = time.time() - start_time
        logger.info(f"Processing time: {process_time:.2f}s")
        logger.info(f"Merging time: {merge_time:.2f}s")

    def process_wrapper(self, idx, mcap_file):
        logger.info(f'Started processing {idx+1}: {mcap_file}')
        test_count = 0
        while True:
            self.process(idx, mcap_file)
            logger.info(f'Completed processing {idx+1}: {mcap_file}')
            break
        

    def process(
            self, 
            idx, # for lerobot dataset merging
            mcap_info, # an element of rawDataList
        ):
        start_time = time.time()
        mcap_path = mcap_info["path"]
        processed_msgs = self.extract(mcap_path)
        head_rgb_timestamps = np.array([self.msg_to_timestamp(msg) for msg in processed_msgs[RGB_HEAD_LEFT_TOPIC]])
        fps = int(np.round(1.0 / np.median(head_rgb_timestamps[1:] - head_rgb_timestamps[:-1])))
        self.fps_dict[str(idx)] = fps
        index_array = np.array([0, len(head_rgb_timestamps)])

        for topic, data in processed_msgs.items():
            if data is None:
                logger.warning(f'Message from {topic} is None!')
                continue
            
            if topic == RGB_HEAD_LEFT_TOPIC:
                bridge = CvBridge()
                head_rgb_images_list = []
                head_images = np.array([bridge.compressed_imgmsg_to_cv2(msg) for msg in data])
                self.shape_of_images["HEAD_LEFT_RGB"] = head_images[0].shape
                for i in range(len(index_array) - 1):
                    head_rgb_images_list.append(head_images[index_array[i]:index_array[i+1]])
                processed_msgs[topic] = head_rgb_images_list

            elif topic in self.RGB_TOPICS:
                bridge = CvBridge()
                wrist_rgb_images_list = []
                first_image_shape = bridge.compressed_imgmsg_to_cv2(data[0]).shape
                if topic == RGB_WRIST_LEFT_TOPIC:
                    self.shape_of_images["WRIST_LEFT_RGB"] = first_image_shape
                if topic == RGB_WRIST_RIGHT_TOPIC:
                    self.shape_of_images["WRIST_RIGHT_RGB"] = first_image_shape
                aligned_wrist_rgb_images = self.align_rgb(head_rgb_timestamps, data)
                for i in range(len(index_array) - 1):
                    wrist_rgb_images_list.append(aligned_wrist_rgb_images[index_array[i]:index_array[i+1]])
                processed_msgs[topic] = wrist_rgb_images_list
                
            elif topic in self.JOINT_TOPICS:
                timestamps = []
                positions = []
                velocities = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    positions.append(list(msg.position))
                    velocities.append(list(msg.velocity))
                interpolated_positions = self.interpolate_1d(head_rgb_timestamps, timestamps, positions)
                try:
                    interpolated_velocities = self.interpolate_1d(head_rgb_timestamps, timestamps, velocities)
                except Exception as e:
                    logger.error(f"Error interpolating velocities for mcap_path={mcap_path}: {e}")
                    raise
                joint_dict_list = list()
                for i in range(len(index_array) - 1):
                    joint_dict = dict()
                    joint_dict['position'] = interpolated_positions[index_array[i]:index_array[i+1]]
                    joint_dict['velocity'] = interpolated_velocities[index_array[i]:index_array[i+1]]
                    joint_dict_list.append(joint_dict)
                processed_msgs[topic] = joint_dict_list
            
            elif topic in self.POSE_TOPICS:
                pose_transforms = []
                pose_timestamps = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    transform_ref = msg.pose
                    pos = [transform_ref.position.x, transform_ref.position.y, transform_ref.position.z]
                    quat = [transform_ref.orientation.x, transform_ref.orientation.y, transform_ref.orientation.z, transform_ref.orientation.w]
                    pose_transforms.append(pos + quat)
                    pose_timestamps.append(timestamp)
                pose_transforms = np.array(pose_transforms)
                pose_timestamps = np.array(pose_timestamps)
                transforms = self.interpolate_transform(head_rgb_timestamps, pose_timestamps, pose_transforms)

                pose_list = []
                for i in range(len(index_array) - 1):
                    pose_list.append(transforms[index_array[i]:index_array[i+1]])
                
                processed_msgs[topic] = pose_list

            elif topic in self.GRIPPER_TOPICS:
                timestamps = []
                positions = []
                gripper_list = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    positions.append([msg.position[0]])
                timestamps = np.array(timestamps)
                positions = np.array(positions)
                if len(positions) > 0:
                    positions = self.interpolate_1d(head_rgb_timestamps, timestamps, positions)
                
                for i in range(len(index_array) - 1):
                    gripper_list.append(positions[index_array[i]:index_array[i+1]])                
                processed_msgs[topic] = gripper_list

            elif topic in self.TWIST_TOPICS:
                timestamps = []
                velocities = []
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    velocities.append([
                        msg.twist.linear.x, 
                        msg.twist.linear.y, 
                        msg.twist.linear.z, 
                        msg.twist.angular.x, 
                        msg.twist.angular.y, 
                        msg.twist.angular.z
                    ])

                timestamps = np.array(timestamps)  # do not set dtype for timestamps, it exceeds the upper bound of fp32
                velocities = np.array(velocities)
                if len(velocities) > 0:
                    velocities = self.interpolate_1d(head_rgb_timestamps, timestamps, velocities)
                
                velocity_list = []
                for i in range(len(index_array) - 1):
                    velocity_list.append(velocities[index_array[i]:index_array[i+1]])
                
                processed_msgs[topic] = velocity_list
            
            elif topic in self.IMU_TOPICS:
                timestamps = []
                imu = []
                imu_transforms = []
                pos = [0.0, 0.0, 0.0]
                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    imu.append([
                        msg.angular_velocity.x, 
                        msg.angular_velocity.y, 
                        msg.angular_velocity.z, 
                        msg.linear_acceleration.x, 
                        msg.linear_acceleration.y, 
                        msg.linear_acceleration.z
                    ])
                    imu_transforms.append(pos + [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

                timestamps = np.array(timestamps)
                imu = np.array(imu)
                imu_transforms = np.array(imu_transforms)
                imu = self.interpolate_1d(head_rgb_timestamps, timestamps, imu)
                transforms = self.interpolate_transform(head_rgb_timestamps, timestamps, imu_transforms)
                
                imu_list = []
                for i in range(len(index_array) - 1):
                    imu_list.append(np.concatenate((transforms[index_array[i]:index_array[i+1], 3:], imu[index_array[i]:index_array[i+1]]), axis=-1))
                
                processed_msgs[topic] = imu_list

            elif topic in self.CONTROL_TOPICS:
                timestamps = []
                positions = []
                velocities = []

                for msg in data:
                    timestamp = self.msg_to_timestamp(msg)
                    timestamps.append(timestamp)
                    positions.append(list(msg.p_des))
                    velocities.append(list(msg.v_des))

                if len(positions) > 0:
                    positions = self.interpolate_1d(head_rgb_timestamps, timestamps, positions)
                
                if len(velocities) > 0:
                    velocities = self.interpolate_1d(head_rgb_timestamps, timestamps, velocities)

                joint_dict_list = list()
                for i in range(len(index_array) - 1):
                    joint_dict = dict()
                    joint_dict['position'] = positions[index_array[i]:index_array[i+1]]
                    joint_dict['velocity'] = velocities[index_array[i]:index_array[i+1]]
                    joint_dict_list.append(joint_dict)
                processed_msgs[topic] = joint_dict_list

        episode = self.create_episode(processed_msgs)
        features = self.create_features(episode[0])
        lerobot_dataset = LeRobotDataset.create(
            repo_id=f'Galaxea/{self.dataset_name}_{idx}',
            features=features,
            robot_type=self.robot_type,
            root=os.path.join(self.output_dir, self.dataset_name) + f'_{idx}',
            fps=fps,
        )

        # get coarse description and quality label from dcTask and labelsStr
        # as qualitySubLabel is used by base quality check
        episode_description = format_shelf_string(self.dataset_name)

        # get fine description and quality label from annotation.text and annotation.actionQualityLabel
        framewise_descriptions = []
        framewise_quality = []
        annotations = mcap_info.get("annotations", None)
        if self.use_translation:
            for annotation in annotations:
                annotation["translated_text"] = self.deepseek_translate_instruction(annotation["text"])
        for timestamp in head_rgb_timestamps:
            annotated = False
            # HACK: due to some mismatching between header timestamp and EDP, 
            # some early frames are not included in annotation interval
            if annotations is None:
                annotations = [dict(startSecond=0, startNanoSecond=0, endSecond=0, endNanoSecond=0)]
            else:
                annotations[0]["startSecond"] = 0
                annotations[0]["startNanoSecond"] = 0
            for cur_annotation in annotations:
                start_timestamp = cur_annotation["startSecond"] + cur_annotation["startNanoSecond"] / 1e9
                end_timestamp = cur_annotation["endSecond"] + cur_annotation["endNanoSecond"] / 1e9
                if timestamp > start_timestamp and timestamp < end_timestamp:
                    if self.use_translation:
                        description = f"{cur_annotation['text']}@{cur_annotation['translated_text']}"
                    else:
                        description = cur_annotation["text"]
                    framewise_descriptions.append(description)
                    framewise_quality.append("qualified" if cur_annotation["actionQualityLabel"] in ["None", "qualified"] else "unqualified")
                    annotated = True
                    break
            if not annotated:
                framewise_descriptions.append("null")
                framewise_quality.append("qualified")
        
        if len(episode) != 0:
            time_start = time.time()
            for frame, description, quality in zip(episode, framewise_descriptions, framewise_quality):
                lerobot_dataset.add_frame(
                    frame=frame,
                    task=episode_description#[episode_description, description, quality],
                )
            time_end = time.time()
            logger.info(f"add_frame time: {time_end - time_start} seconds")
            time_start = time.time()
            lerobot_dataset.save_episode()
            time_end = time.time()
            logger.info(f"save_episode time: {time_end - time_start} seconds")
            return (mcap_path, "success")
        else:
            return (mcap_path, "failed")

    
    def create_episode(self, processed_dataset):
        episode = []
        for i in range(len(processed_dataset[RGB_HEAD_LEFT_TOPIC][0])):
            frame = {}
            frame["observation.images.head_rgb"] = processed_dataset[RGB_HEAD_LEFT_TOPIC][0][i]
            frame["observation.images.head_right_rgb"] = processed_dataset[RGB_HEAD_RIGHT_TOPIC][0][i]
            frame["observation.images.left_wrist_rgb"] = processed_dataset[self.RGB_WRIST_LEFT_TOPIC][0][i]
            frame["observation.images.right_wrist_rgb"] = processed_dataset[self.RGB_WRIST_RIGHT_TOPIC][0][i]
            
            frame["observation.state.left_arm"] = processed_dataset[JOINT_OBS_LEFT_TOPIC][0]["position"][i][0: self.arm_dof]
            frame["observation.state.left_arm.velocities"] = processed_dataset[JOINT_OBS_LEFT_TOPIC][0]["velocity"][i][0: self.arm_dof]
            frame["observation.state.right_arm"] = processed_dataset[JOINT_OBS_RIGHT_TOPIC][0]["position"][i][0: self.arm_dof]
            frame["observation.state.right_arm.velocities"] = processed_dataset[JOINT_OBS_RIGHT_TOPIC][0]["velocity"][i][0: self.arm_dof]
            frame["observation.state.left_gripper"] = processed_dataset[GRIPPER_OBS_LEFT_TOPIC][0]["position"][i]
            frame["observation.state.right_gripper"] = processed_dataset[GRIPPER_OBS_RIGHT_TOPIC][0]["position"][i]
            frame["observation.state.chassis.imu"] = processed_dataset[CHASSIS_IMU_TOPIC][0][i]
            frame["observation.state.chassis"] = processed_dataset[CHASSIS_OBS_TOPIC][0]['position'][i][0:3]
            # FIXME: The feedback for the chassis provides a 6-dim velocity, 
            # but only the first 3 dims are valid. The last 3 dims do not change.
            frame["observation.state.chassis.velocities"] = processed_dataset[CHASSIS_OBS_TOPIC][0]['velocity'][i][0:3]
            frame["observation.state.torso"] = processed_dataset[TORSO_OBS_TOPIC][0]["position"][i]
            frame["observation.state.torso.velocities"] = processed_dataset[TORSO_OBS_TOPIC][0]["velocity"][i]
            frame["observation.state.left_ee_pose"] = processed_dataset[EE_POSE_OBS_LEFT_TOPIC][0][i]
            frame["observation.state.right_ee_pose"] = processed_dataset[EE_POSE_OBS_RIGHT_TOPIC][0][i]
            
            if self.robot_type == "r1pro":
                frame["action.left_ee_pose"] = processed_dataset[EE_POSE_ACTION_LEFT_TOPIC][0][i]
                frame["action.right_ee_pose"] = processed_dataset[EE_POSE_ACTION_RIGHT_TOPIC][0][i]
            
            frame["action.left_gripper"] = processed_dataset[GRIPPER_ACTION_LEFT_TOPIC][0][i]
            frame["action.right_gripper"] = processed_dataset[GRIPPER_ACTION_RIGHT_TOPIC][0][i]
            frame["action.left_arm"] = processed_dataset[JOINT_ACTION_LEFT_TOPIC][0]["position"][i]
            frame["action.right_arm"] = processed_dataset[JOINT_ACTION_RIGHT_TOPIC][0]["position"][i]
            frame["action.chassis.velocities"] = processed_dataset[CHASSIS_ACTION_TOPIC][0][i]
            
            # only R1 Pro with whole-body control has torso joint action, while R1 Lite still uses torso speed control
            if self.robot_type == "r1pro" and len(processed_dataset[TORSO_ACTION_TOPIC][0]["position"]) > 0:
                frame["action.torso"] = processed_dataset[TORSO_ACTION_TOPIC][0]["position"][i]
            if self.robot_type == "r1lite" and len(processed_dataset[TORSO_ACTION_SPEED_TOPIC][0]) > 0:
                frame["action.torso.velocities"] = processed_dataset[TORSO_ACTION_SPEED_TOPIC][0][i]

            episode.append(frame)
        
        return episode
    
    def upload(self, ):
        if self.use_compression:
            def add_to_tar(tar, file_path):
                tar.add(file_path, arcname=os.path.basename(file_path))
            files = [os.path.join(self.output_dir, self.dataset_name), 
                    os.path.join(self.output_dir, "training_data_set_meta.json")]
            output_tar = os.path.join(self.output_dir, f"{self.dataset_name}.tar.gz")
            with tarfile.open(output_tar, "w:gz") as tar:
                for file in files:
                    try:
                        add_to_tar(tar, file)
                    except Exception as e:
                        print(f"Error adding {file} to tar: {e}")
            shutil.move(output_tar, self.output_dir)
            
    def merge_subdataset(self,):
        subdatasets = []
        for entry in os.scandir(self.output_dir):
            if entry.is_dir():
                full_path = entry.path
                subdatasets.append(full_path)
        subdatasets = sorted(subdatasets)
        logger.info(f'Start merging:\n {subdatasets}\n To: {self.output_dir}/{self.dataset_name}')

        sum_fps = 0
        count = 0
        for key, fps in self.fps_dict.items():
            sum_fps += fps
            count += 1
        mean_fps = sum_fps // count if count > 0 else 15

        merge_datasets(
            source_folders=subdatasets,
            output_folder=os.path.join(self.output_dir, self.dataset_name),
            default_fps=mean_fps, 
        )
    
    def align_rgb(self, head_rgb_timestamps, source_data):
        source_timestamps = []
        source_images = []
        bridge = CvBridge()
        for msg in source_data:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            source_timestamps.append(timestamp)
            source_images.append(bridge.compressed_imgmsg_to_cv2(msg))
        
        source_timestamps = np.array(source_timestamps)
        source_images = np.array(source_images)
        
        closest_indices = np.abs(source_timestamps[:, None] - head_rgb_timestamps).argmin(axis=0)
        source_images = np.array(source_images)
        aligned_images = source_images[closest_indices]

        return aligned_images

    def interpolate_1d(self, target_timestamps, source_timestamps, source_values):
        noise_topic_threshold = 0.2
        if len(source_timestamps) < noise_topic_threshold * len(target_timestamps):
            return source_values
        
        f = interp1d(source_timestamps, source_values, kind='linear', axis=0, bounds_error=False, fill_value=(source_values[0], source_values[-1]))
        interpolated_states = f(target_timestamps)

        return interpolated_states
        
    def interpolate_transform(self, target_timestamps, source_timestamps, source_values):
        positions = source_values[:, :3]
        quaternions = [pyQuaternion(source_values[i, 6], source_values[i, 3], source_values[i, 4], source_values[i, 5]) for i in range(source_values.shape[0])]

        target_values = np.empty((len(target_timestamps), 7))

        # Use numpy's searchsorted to find indices for interpolation
        indices = np.searchsorted(source_timestamps, target_timestamps, side='right')

        # Ensure indices are within valid range
        indices = np.clip(indices, 1, len(source_timestamps) - 1)

        t0 = source_timestamps[indices - 1]
        t1 = source_timestamps[indices]

        pos0 = positions[indices - 1]
        pos1 = positions[indices]

        quat0 = np.array([quaternions[i - 1] for i in indices])
        quat1 = np.array([quaternions[i] for i in indices])

        # Calculate interpolation factors
        t_interp = (target_timestamps - t0) / (t1 - t0)

        target_values[:, :3] = pos0 + np.expand_dims(t_interp, axis=1) * (pos1 - pos0)
        
        interpolated_quats = [pyQuaternion.slerp(quat0[i], quat1[i], t_interp[i]).normalised for i in range(len(t_interp))]
        target_values[:, 3:] = np.array([[q.x, q.y, q.z, q.w] for q in interpolated_quats])

        # For timestamps out of the source range
        target_values[target_timestamps <= source_timestamps[0], :3] = positions[0]
        target_values[target_timestamps <= source_timestamps[0], 3:] = [quaternions[0].x, quaternions[0].y, quaternions[0].z, quaternions[0].w]
        
        target_values[target_timestamps >= source_timestamps[-1], :3] = positions[-1]
        target_values[target_timestamps >= source_timestamps[-1], 3:] = [quaternions[-1].x, quaternions[-1].y, quaternions[-1].z, quaternions[-1].w]

        return target_values
    
    def register_quat(self, a):
        a_np = np.array(a)
        quat = a_np[:, 3: 7]
        
        quat_couple_number = 2
        dot = np.diag(np.dot(quat[0: -1], quat[1:].T))
        
        shift_indices = np.where(dot < 0)[0] + 1
        if len(shift_indices) % quat_couple_number == 1:
            shift_indices = np.append(shift_indices, len(quat))
        fixed_quat = quat.copy()
        for i in range(len(shift_indices) // quat_couple_number):
            fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]] = -fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]]
        assert fixed_quat.shape == quat.shape
        a_np[:, 3: 7] = fixed_quat
        a[...] = a_np

    def deepseek_translate_instruction(self, text, temperature=0.0, max_tokens=200):
        # from openai import OpenAI
        from volcenginesdkarkruntime import Ark

        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        client = Ark(api_key=deepseek_api_key)

        cnt = 0
        while cnt < 5:
            try:
                response = client.chat.completions.create(
                    model="deepseek-v3-1-250821",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的机器人指令翻译专家。"
                        },
                        {
                            "role": "user",
                            "content": f"请将以下中文机器人动作指令翻译成准确简洁、明确的英文，使用祈使句格式，不要添加任何解释：\n\n{text}"
                        }
                    ],
                    thinking={
                        "type": "disabled"
                    },
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content
                text = text.replace("\n", " ")
                text = " ".join(text.split())
                return text
            except Exception as e:
                cnt += 1
                print(f'Failed to translate, {e}, retry {cnt}')
                continue
        return None

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ROS to LeRobot data')
    parser.add_argument('--input_dir', required=True, type=str, help='where ROS  are saved')
    parser.add_argument('--output_dir', required=True, type=str, help='where LeRobot data are saved')
    parser.add_argument('--robot_type', required=True, choices=['R1Pro', 'R1', 'R1Lite'], help='robot type')
    parser.add_argument('--dataset_name', required=True, type=str, help='dataset_name')

    return parser.parse_args()

def search_rosbags(input_dir: str):
    path = Path(input_dir)
    try:
        bag_files = list(path.rglob('*.bag'))
    except Exception as e:
        print(f"Error searching for bag files: {e}")
    if len(bag_files) == 0:
        bag_files = list(path.rglob('*.mcap'))
    return sorted([str(file.absolute()) for file in bag_files])

def get_raw_data_meta_from_args():
    args = get_args()
    dataset_name = args.dataset_name
    robot_type = args.robot_type
    output_dir = args.output_dir
    data_path_list = search_rosbags(args.input_dir)

    raw_data_meta_path = os.path.join(args.input_dir, "raw_data_meta.json")
    if os.path.exists(raw_data_meta_path):
        with open(raw_data_meta_path) as f:
            raw_data_meta_json = json.load(f)
            raw_data_meta_json = dict(data=raw_data_meta_json)
            raw_data_meta_json["data"]["rawDataSetName"] = dataset_name
    else:
        raw_data_meta_dict = dict()
        raw_data_meta_dict["rawDataSetName"] = dataset_name
        raw_data_meta_dict["rawDataList"] = list()
        for data_path in data_path_list:
            data_meta = dict()
            data_meta['name'] = os.path.basename(data_path)
            data_meta['path'] = data_path
            data_meta['robotType'] = robot_type
            raw_data_meta_dict["rawDataList"].append(data_meta)

        raw_data_meta_json = dict(data=raw_data_meta_dict)
    return (raw_data_meta_json, output_dir)



def format_shelf_string(s):
    parts = s.split('_')
    filtered_parts = [part for part in parts if not part.isdigit()]
    final_parts = [part for part in filtered_parts if not part.isdigit() or len(part) < 8]
    result = ' '.join(final_parts).lower()
    return result

if __name__ == '__main__':
    raw_data_meta_json, cache_dir, output_dir = get_raw_data_meta()

    mcaps_dict = raw_data_meta_json['data']
    dataset_name = mcaps_dict['rawDataSetName']
    robot_type = mcaps_dict['rawDataList'][0]['robotType'].lower()
    sample_mcap_path = mcaps_dict["rawDataList"][0]["path"]
    bag_type = mcaps_dict['rawDataList'][0]['name'].split('.')[-1]

    if bag_type == 'bag':
        USE_ROS1 = True
    else:
        USE_ROS1 = False

    if not USE_ROS1:
        import rclpy
        from rclpy.serialization import deserialize_message
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        logger.info(f"Use ROS1")
    else:
        import rosbag
        logger.info(f"Use ROS2")

    if not USE_ROS1:
        rclpy.init()

    if robot_type not in ['r1pro', 'r1', 'r1lite']:
        logger.error(f'Unknown robot type: {robot_type}')
        exit(1)

    try:
        training_data_set_meta_file = os.path.join(output_dir, 'training_data_set_meta.json')
        with open(training_data_set_meta_file, "w", encoding="utf-8") as f:
            json.dump(mcaps_dict, f, indent=4)
    except Exception as e:
        logger.error(f'Error writing {training_data_set_meta_file}, {e}')

    data_converter = DataConverter(
        robot_type, 
        sample_mcap_path, 
        dataset_name, 
        output_dir, 
        use_ros1=USE_ROS1, 
        save_video=SAVE_VIDEO, 
        use_h264=USE_H264, 
        use_compression=USE_COMPRESSION, 
        is_compute_episode_stats_image=IS_COMPUTE_EPISODE_STATS_IMAGE,
        max_processes=MAX_PROCESSES,
        use_translation=USE_TRANSLATION
    )

    # 2. process messages.
    data_converter.process_all(mcaps_dict)

    data_converter.upload()

    if not USE_ROS1:    
        rclpy.shutdown()
