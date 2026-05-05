import logging
import threading
import queue

logger = logging.getLogger(__name__)

try:
    from adafruit_servokit import ServoKit
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logger.warning("adafruit_servokit not found. Running in SIMULATION mode.")

class MotorController:

    # Initialization for the MotorController class, setting up the frame
    # dimensions and motor control parameters
    def __init__(self, frame_h, frame_w):
        """ Initialize Motor Controller. """
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.base_w = 1280.0
        self.base_h = 720.0

        # calc center of the screen once and store it for later use
        self.screen_center_x = frame_w / 2.0
        self.screen_center_y = frame_h / 2.0

        # Only keep the most recent motor command, delete older ones
        self.motor_queue = queue.Queue(maxsize=1) 

        # set up thread control
        self.running = threading.Event()
        self.thread = None

        # set starting angles for the motors to be in the middle position (90 degrees)
        self.current_pan = 90 
        self.current_tilt = 25 

        # tuning variables
        # Scale tuning from known-good 1280x720 behavior to current resolution.
        self.threshold_x = 45.0 * (self.frame_w / self.base_w)
        self.threshold_y = 45.0 * (self.frame_h / self.base_h)
        self.kp_pan = 0.015 * (self.base_w / self.frame_w)
        self.kp_tilt = 0.015 * (self.base_h / self.frame_h)
        self.max_step_deg = 1.2
        self.min_step_deg = 0.15

        self.kit = None
        self.hardware_enabled = False

        # initialize hardware if available
        if HARDWARE_AVAILABLE:
            try:
                self.kit = ServoKit(channels=16)
                self.kit.servo[0].angle = self.current_pan
                self.kit.servo[1].angle = self.current_tilt
                self.hardware_enabled = True
                logger.info("Adafruit ServoKit initialized.")
            except Exception as err:
                logger.warning(
                    "Servo hardware init failed (%s). Falling back to simulation mode.",
                    err,
                )
        else:
            logger.warning("Hardware not available. MotorController will run in simulation mode.")

    def start(self) -> None:
        """Starts the background motor thread."""
        self.running.set()
        self.thread = threading.Thread(
            target=self._motor_worker_loop, 
            daemon=True, 
            name="motor-loop"
        )
        self.thread.start()
        logger.info("Motor Controller thread started.")

    def stop(self) -> None:
        """Stops the motor thread safely."""
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=1.0)
            
        logger.info("Motor Controller thread stopped.")

    def update_target(self, obj_x: float, obj_y: float) -> None:
        """
        Called by the fast AI thread. Drops old data if the motor is busy.
        """
        if self.motor_queue.full():
            try:
                self.motor_queue.get_nowait() # throw away old data

                logger.debug("Motor busy: Dropping older tracking frame.")
            except queue.Empty:
                pass
                
        # Put the newest center coordinates in the queue
        self.motor_queue.put((obj_x, obj_y))

    def _motor_worker_loop(self) -> None:
        """
        The background loop that constantly consumes queue data and moves hardware.
        """
        while self.running.is_set():
            try:
                # Wait for new coordinates from YOLO
                obj_x, obj_y = self.motor_queue.get(timeout=0.1)
                
                # Calculate the pixel difference (error)
                diff_x = obj_x - self.screen_center_x
                diff_y = obj_y - self.screen_center_y
                
                # Check deadzones and move
                if abs(diff_x) > self.threshold_x:
                    self._pan_motor_adj(diff_x)
                    
                if abs(diff_y) > self.threshold_y:
                    self._tilt_motor_adj(diff_y)
                    
            except queue.Empty:
                # No target seen recently, do nothing
                continue

    def _pan_motor_adj(self, diff_x: float) -> None:
        """Translates pixel error into a smooth Pan movement."""
        # Calculate angle step
        angle_step = diff_x * self.kp_pan
        angle_step = max(-self.max_step_deg, min(self.max_step_deg, angle_step))
        if abs(angle_step) < self.min_step_deg:
            return
        
        # update current angle (Change to += if motor turns the wrong way)
        self.current_pan -= angle_step 
        
        # Clamp between physical servo limits (0 to 180 degrees)
        self.current_pan = max(0.0, min(180.0, self.current_pan))
        
        if self.hardware_enabled and self.kit is not None:
            try:
                self.kit.servo[0].angle = self.current_pan 
            except OSError: # OSError usually means a hardware disconnection
                logger.error("CRITICAL: Lost communication with PCA9685 board over I2C!")
        else:
            logger.debug(f"[SIM] Panning to: {self.current_pan:.1f}° (Step: {angle_step:.2f}°)")

    def _tilt_motor_adj(self, diff_y: float) -> None:
        """Translates pixel error into a smooth Tilt movement."""
        angle_step = diff_y * self.kp_tilt
        angle_step = max(-self.max_step_deg, min(self.max_step_deg, angle_step))
        if abs(angle_step) < self.min_step_deg:
            return
        
        # Invert tilt update so positive screen error moves camera downward.
        self.current_tilt -= angle_step 
        
        self.current_tilt = max(0.0, min(180.0, self.current_tilt))
        
        if self.hardware_enabled and self.kit is not None:
            try:
                self.kit.servo[1].angle = self.current_tilt
            except OSError: # hardware disconnection
                logger.error("CRITICAL: Lost communication with PCA9685 board over I2C!")
        else:
            logger.debug(f"[SIM] Tilting to: {self.current_tilt:.1f}° (Step: {angle_step:.2f}°)")

    def reset_to_center(self) -> None:
        """Instantly snaps both motors back to the 90-degree center position."""
        self.current_pan = 90.0
        self.current_tilt = 90.0
        
        if self.hardware_enabled and self.kit is not None:
            try:
                self.kit.servo[0].angle = self.current_pan
                self.kit.servo[1].angle = self.current_tilt
                logger.info("Motors reset to center (90.0°).")
            except OSError:
                logger.error("CRITICAL: Lost communication with PCA9685 board over I2C!")
        else:
            logger.info("[SIM] Resetting motors to center (90.0°).")

    def manual_move(self, pan_step: float, tilt_step: float) -> None:
        """
        Manually nudges the motors by a specific degree amount. 
        Useful for keyboard, API, or joystick control.
        """
        # Add the steps to the current angles
        self.current_pan += pan_step
        self.current_tilt += tilt_step
        
        # Hardware safety clamp (0 to 180 degrees)
        self.current_pan = max(0.0, min(180.0, self.current_pan))
        self.current_tilt = max(0.0, min(180.0, self.current_tilt))
        
        if self.hardware_enabled and self.kit is not None:
            try:
                self.kit.servo[0].angle = self.current_pan
                self.kit.servo[1].angle = self.current_tilt
            except OSError:
                logger.error("CRITICAL: Lost communication with PCA9685 board over I2C!")
        else:
            logger.debug(f"[SIM] Manual move -> Pan: {self.current_pan:.1f}°, Tilt: {self.current_tilt:.1f}°")
