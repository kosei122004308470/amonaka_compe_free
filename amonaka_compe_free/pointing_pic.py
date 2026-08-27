import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraCapture(Node):

    def __init__(
        self,
        image_topic="/image_raw",
        node_name="camera_capture",
    ):
        super().__init__(node_name)

        self.bridge = CvBridge()

        # 最新のROS Image
        self.latest_image = None

        # カメラ画像を常時subscribe
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            10,
        )

        self.get_logger().info(
            f"CameraCapture started: {image_topic}"
        )

    def _image_callback(self, msg: Image):
        """
        カメラから受信した最新画像を保持する。

        JPEG化はここでは行わない。
        """
        self.latest_image = msg

    def capture(self, quality=90):
        """
        capture()を呼び出したタイミングで
        最新のカメラ画像をJPEG化する。

        Returns:
            bytes | None:
                JPEGデータ
        """

        # ------------------------------------------------
        # ROSのcallbackを処理
        # ------------------------------------------------
        #
        # ここで最新の/image_rawを受信する
        #
        rclpy.spin_once(
            self,
            timeout_sec=0.05
        )

        # まだ画像を受信していない場合
        if self.latest_image is None:
            self.get_logger().warn(
                "カメラ画像をまだ受信していません。"
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
