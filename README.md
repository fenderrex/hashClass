Welcome to your interactive grid world! This tool helps you explore how a car finds its way to a goal while avoiding obstacles — all while looking great on the map.

🎨 The Drawing Area
There’s a big canvas — like your playmat — where everything happens.

It’s split into grid squares. You can make the squares bigger or smaller using the grid size box.

🚗 Start and Goal
A red circle is your car (start point).

A green circle is your castle (goal point).

You can drag them anywhere to set new locations.

⛰️ Obstacles
Black circles are rocks that block the car.

Click “Randomize Obstacles” to drop 8 new rocks in random spots.

📍 Finding the Path
Click “Run” and the computer will:

Look at the grid

Avoid rocks

Find a safe route from car to castle

✏️ Path Smoothing – “Magic Crayons”
After finding a basic step-by-step route, the simulator makes it prettier using smoothing algorithms:

A + Earcut* → Simple path with triangles

+ Bézier → Adds curved lines

+ Chaikin → Makes it soft and flowing

Full Pipeline → All of the above together!

Choose your smoothing style in the dropdown.

🧠 Debug Info
The gray debug box shows behind-the-scenes stats:

How many squares were checked

How smooth the path became

And more info to track performance

🖼️ Map Overlays
The tool uses a map overlay called pathconvergance.html?key=YOUR_KEY_HERE, where key is your doodle map key.

This page helps match real-world boundaries with paths.

New map boundaries are included in the name and are used directly by pathconvergance.html.

🤖 Simulation & SegNet
Simulation photos and predictions are handled automatically.

SegNet is used for segmentation (like roads or sidewalks).

You can run predictions with the file 13.py using SegNet files.

🖥️ Running It on Windows
To get started easily:

Run the included file: install_and_start.bat

It will:

Create a virtual Python environment on your Desktop

Install everything you need

Launch the simulator with the image abbb1.png

This tool is like asking your computer:

“What’s the smartest way to get from here to there, while dodging rocks and looking smooth doing it?”

And it draws that path just for you! 🎉
