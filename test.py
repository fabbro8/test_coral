from periphery import GPIO

# Open GPIO's
gpio6 = GPIO(6, "in")
gpio7 = GPIO(7, "in")
gpio8 = GPIO(8, "in")
gpio138 = GPIO(138, "in")
gpio140 = GPIO(140, "in")
gpio141 = GPIO(141, "in")
gpio73 = GPIO(73, "out")
gpio77 = GPIO(77, "out")

GPIO.EdgeEvent(gpio6)
