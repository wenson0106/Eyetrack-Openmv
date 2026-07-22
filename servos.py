from machine import Pin, PWM
import time

class Servo:
    """
    A class responsible for controlling servos via the OpenMV board.
    """

    def __init__(self):
        """
        Initialise the servo object and sets the tuning coefficients.
        """
        # Servo tuning coefficients; EDIT these values as required.
        self.pan_angle_corr = 0
        self.left_zero = 0.05
        self.right_zero = -0.1

        # Define servo pin names for the servo shield. EDIT these values as required.
        # NOTE: P9/P10 share one FlexPWM submodule (and P10 doubles as the camera
        # frame-sync line), so they are NOT independent - avoid pairing two servos
        # on P9+P10. Pins below are spread across separate submodules instead.
        self.pan_id = 'P10'
        self.left_id = 'P8'
        self.right_id = 'P7'

        # Set up servo angle limits
        self.degrees = 180
        self.min_deg = -self.degrees/2
        self.max_deg = self.degrees/2

        self.curr_l_speed = 0
        self.curr_r_speed = 0
        self.pan_pos = 0

        self.freq = 50

        # Pulse width range (microseconds) for the PWM signal.
        self.min_us = 700
        self.max_us = 2300
        self.mid_us = (self.min_us + self.max_us) / 2
        self.span_us = (self.max_us - self.min_us)

        # Initialise a PWM channel per servo (newer servo shields drive servos
        # directly over GPIO rather than via a PCA9685 I2C chip).
        self.pan_pwm = PWM(Pin(self.pan_id), freq=self.freq)
        self.left_pwm = PWM(Pin(self.left_id), freq=self.freq)
        self.right_pwm = PWM(Pin(self.right_id), freq=self.freq)

    def set_differential_drive(self, speed: float, bias: float) -> None:
        """
        Set speeds for a differential drive robot using a speed coefficient and a steering bias.

        Args:
            speed_coeff (float): Overall speed coefficient of the robot (0 to 1).
            steering_bias (float): Steering bias for the robot (-1 to 1).
        """
        # Validate input ranges
        speed = max(min(speed, 1), 0)
        bias = max(min(bias, 1), -1)

        # Calculate individual wheel speeds
        left_speed = speed * (1 - bias)
        right_speed = speed * (1 + bias)

        # Normalize speeds if they exceed 1
        max_speed = max(abs(left_speed), abs(right_speed))
        if max_speed > 1:
            left_speed /= max_speed
            right_speed /= max_speed

        # Set the speeds
        self.set_speed(left_speed, right_speed)


    def set_angle(self, angle: float) -> float:
        """
        Set the pan servo to a specific angle (in degrees).

        Args:
            angle (float): Desired angle (deg) for the camera pan servo.

        Returns:
            float: Corrected angle (deg) of the camera pan servo.
        """
        # Correct for off centre angle
        self.pan_pos = angle

        angle = self.pan_angle_corr + angle

        # Constraint angle to limits
        angle = max(min(angle, self.max_deg), self.min_deg)

        # Compute pulse width (us) for the PWM signal
        pulse_us = self.mid_us + ( self.span_us * (angle / self.degrees) )

        # Set duty and send PWM signal
        self.pan_pwm.duty_ns(int(pulse_us * 1000))

        return self.pan_pos


    def set_speed(self, l_speed: float, r_speed: float) -> None:
        """
        Control the speed of the left and right wheel servos.

        Args:
            l_speed (float): Speed to set left wheel servo to (-1~1).\n
            r_speed (float): Speed to set right wheel servo to (-1~1).
        """

        # Constraint speeds to limits
        l_speed = max(min(l_speed, 1), -1)
        r_speed = max(min(r_speed, 1), -1)
        self.curr_l_speed = l_speed
        self.curr_r_speed = r_speed

        # Convert speed to pulse width (us)
        l_us = self.mid_us + (self.span_us / 2 * (self.curr_l_speed + self.left_zero))
        r_us = self.mid_us - (self.span_us / 2 * (self.curr_r_speed + self.right_zero))

        # Ensure pulse width values are within the valid range
        l_us = max(min(l_us, self.max_us), self.min_us)
        r_us = max(min(r_us, self.max_us), self.min_us)

        # Set duty and send PWM signal
        self.left_pwm.duty_ns(int(l_us * 1000))
        self.right_pwm.duty_ns(int(r_us * 1000))

        return

    def release(self, pwm: PWM) -> None:
        """
        Simple servo release method

        Args:
            pwm (PWM): Servo PWM channel to reset (e.g. self.pan_pwm).
        """
        pwm.duty_ns(0)


    def release_all(self) -> None:
        """
        Release all servos.
        """
        for pwm in (self.pan_pwm, self.left_pwm, self.right_pwm):
            self.release(pwm)


    def soft_reset(self) -> None:
        """
        Method to reset the servos to default and print a delay prompt.
        """
        # Reset all servo shield pins
        self.release_all()

        # Reset pan to centre
        self.set_angle(0)

        # Print delay prompt
        for i in range(3, 0, -1):
            print(f"{i} seconds remaining.")
            time.sleep_ms(1000)

        print("___Running Code___")
