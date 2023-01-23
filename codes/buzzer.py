import RPi.GPIO as GPIO
from time import sleep

# disable warning
GPIO.setwarnings(False)

# Select GPIO mode
GPIO.setmode(GPIO.BCM)

# Set buzzer - pin 23 as output
buzzer = 23
GPIO.setup(buzzer, GPIO.OUT)

def buzzerAlert(stop):
    # Run forever loop
    while True:
        if stop():
            GPIO.output(buzzer, GPIO.LOW)
            break
        
        GPIO.output(buzzer, GPIO.HIGH)
        sleep(0.5)
        
        GPIO.output(buzzer, GPIO.LOW)
        sleep(0.5)
