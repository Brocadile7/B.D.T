Hello! Thank you for choosing to read this read me!

In this project I am making a computer vision AI tracking turret.

For this project used python! (only language im good at right now
I know a little C+ but not much)

Currently I have made most all of the code for the computer vison 
and it tracks people using yolov8n and aims the crossbar at around the chest
(To avoid injury and prevent head and eye shots when I add hardware)
and I can toggle through Armed and Disarmed when I was coding some errors
I ran into were the body tracking kept only tracking the 
body and not the head so to fix that I fixed up the code to include the whole body but it still 
sometimes will take the head out of the picture that is another reason I added the red hat filter.
Also the crossbar was aiming too low (we don't want it to aim their!) and was changing depending on how
far away I was from the camera so I am planning on adding some more code to lower the crossbar a little
every time it gets a little closer and the same for when i am walking away from it.

Controls are
A = Armed/Disarmed
Q = Quit
(must be uppercase)
also, a tip for if you use this code make sure you have yolo installed and 
when you run it if it does not change Armed/Disarmed and you can't quit, its probably 
because you clicked something besides the screen that pops up when you run the code,
To fix this, just click the screen that has the computer vison playing and that should fix it.

Thankyou!
This is one of my first hardware projects
