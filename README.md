# RoboCar_RaspberryPi
As the title, this is the project I currently work on Raspberry Pi and a RoboCar.

1. Raspberry Pi is acting as a controller, and RoboCar is the actuator.
2. Providing RoboCar is in idle mode in the beginning, until it detects some object entering. 
3. When new object is detected, RoboCar will go towards the object.

So this project contains mainly two parts:

1. Detect new object: actually the common motion detecting problem, with background remains unchanged, new object is detected by taking the difference between the previous and current frame.
2. Moving towards the object: the main challenge of this project, this time the background is changing, and the Raspberry Pi will have to keep the lock on detected object.

At present, Raspberry Pi can detect new object and keep it tracked. But sometimes with sudden moves, the object still got lost. Still trying to improve.
