import time

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraCapture(Node):

    def __init__(
        self,
        image_topic="/image_raw",
        node_name="camera_capture",
    ):
        # launch の __node remap をこの補助ノードへ適用しないようにします。
        super().__init__(node_name, use_global_arguments=False)

        self.bridge = CvBridge()

        # 最新のROS Image
        self.latest_image = None

        # カメラ画像を常時subscribe
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"CameraCapture started: {image_topic}"
        )

    def _image_callback(self, msg: Image):
        """カメラから受信した最新画像を保持する."""
        self.latest_image = msg

    def capture(self, quality=90, timeout_sec=5.0):
        """
        最新のカメラ画像をJPEG化する.

        Returns
        -------
        bytes | None
            JPEGデータ.

        """
        # ------------------------------------------------
        # ROSのcallbackを処理
        # ------------------------------------------------
        #
        # ここで設定された画像トピックの最新フレームを受信する
        #
        deadline = time.monotonic() + timeout_sec
        while self.latest_image is None and rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(
                self,
                timeout_sec=min(0.1, remaining),
            )

        # まだ画像を受信していない場合
        if self.latest_image is None:
            self.get_logger().warn(
                f"{timeout_sec:.1f}秒以内にカメラ画像を受信できませんでした。"
            )
            return None

        try:
            # ------------------------------------------------
            # ROS Image -> OpenCV
            # ------------------------------------------------

            frame = self.bridge.imgmsg_to_cv2(
                self.latest_image,
                desired_encoding="bgr8",
            )

            # ------------------------------------------------
            # OpenCV -> JPEG
            # ------------------------------------------------

            success, encoded_image = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    quality,
                ],
            )

            if not success:
                self.get_logger().error(
                    "JPEG変換に失敗しました。"
                )
                return None

            # JPEG bytesを返す
            jpeg_data = encoded_image.tobytes()

            self.get_logger().info(
                f"Captured JPEG: {len(jpeg_data)} bytes"
            )

            return jpeg_data

        except Exception as e:

            self.get_logger().error(
                f"Capture failed: {e}"
            )

            return None
