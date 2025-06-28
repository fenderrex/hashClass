# 🗺️ PathConvergence Web App

**PathConvergence** is a browser-based tool that finds and visualizes the optimal path between two points on a map image. It uses transparency-based obstacle detection and path smoothing algorithms to generate clean, realistic paths over grid-aligned environments.

This project is ideal for experimenting with A* pathfinding, image-based navigation, segmentation overlays, and simulation of real-world environments like roads, sidewalks, and custom regions.

---

## 🚀 Demo Preview

> “What’s the smartest way across this image map, without walking into invisible (transparent) zones?”

**PathConvergence** answers that question with a clean, colorful route.

---

## 🔧 Features

### 🖼️ Image-Based Map Overlays
- Upload any **PNG** or similar image to serve as a map.
- The **alpha channel (transparency)** determines walkability:
  - Alpha below threshold = ⛔ not walkable
  - Alpha above threshold = ✅ walkable
- Adjustable threshold slider to fine-tune sensitivity.

### 🟢 Start and 🔴 Goal Placement
- Click anywhere on the canvas to reposition start and goal.
- Left or right click sets each point interchangeably.

### ⛰️ Obstacle Detection
- Transparent zones from the uploaded image are treated as impassable.
- Optional **“Randomize Obstacles”** button creates test barriers (rocks).

### ✏️ Path Smoothing Modes
Enhance the visual quality and realism of your path:
- `A* + Earcut` – Fast triangle-meshed path
- `+ Bézier` – Adds curved transitions to reduce sharp corners
- `+ Chaikin` – Further softens curves into flowing lines
- `Full Pipeline` – Combines all of the above for a polished look

### 📍 Map Overlays via Key
- Load maps dynamically using: Upload B/W Map: or the file picked in the menu. ## 🧭 UI Control Panel Guide

This section explains the controls used for managing map overlays, opacity, drawing, and exports in the PathConvergence web app.

---

### 🔘 Buttons and Inputs

#### `Hide Panel` / `Show Panel`
- Toggle visibility of this control panel itself.

#### `Hide Base Map` / `Show Base Map`
- Toggle the visibility of the main base map image (useful when comparing overlays or mask outputs).

#### `Pick Random Start & Goal`
- Automatically places start (red) and goal (green) markers at random walkable points.

#### `Choose File`
- Upload a custom image overlay (e.g., segmented mask or transparency map).
- The selected image is layered over the base map for pixel sampling or masking.
- based on OSM data exported from georefv1
---

### 🌈 Opacity

#### `Opacity Slider`
- Adjusts the transparency of the uploaded overlay image.
- Helps you blend the overlay with the base map to visually inspect walkability.

#### `Full Opacity`
- Forces the overlay image to full opacity (ignores slider).
- Helpful for clean binary visual comparisons.

---

### 🗺️ Geo Coordinates

Set real-world bounds (optional, for georeferenced exports):

- **South Latitude**: Bottom boundary of the map
- **West Longitude**: Left boundary
- **North Latitude**: Top boundary
- **East Longitude**: Right boundary

These values are required for accurate **GeoJSON** exports.

---

### ✍️ Drawing Tools

#### `Read Pixels`
- Reads pixel data from the current overlay.
- Used to extract walkable areas or define mask regions based on color/alpha.

#### `Start Drawing`
- Enables freehand drawing mode on the canvas.
- You can manually trace or mark walkable/blocked regions.

---

### 📤 Export Options

#### `Export GeoJSON`
- Outputs a vectorized version of the drawn or detected regions.
- Requires geolocation bounds to be set.

#### `Export PNG`
- Saves a raster image of the current canvas.
- Includes all overlays, masks, and drawings as shown on screen.

---

### 🎨 Mask Color

- Displays the **current mask color** used during drawing.
- Editable via color picker (click the colored box).
- Drawn regions will be filled using this color.

---

### Example Workflow

1. Upload your base map (e.g. a PNG showing sidewalks).
2. Optionally upload a segmentation mask or overlay.
3. Adjust opacity to see overlap clearly.
4. Use `Read Pixels` or `Start Drawing` to trace or define regions.
5. Export to GeoJSON if working with real-world coordinates, or export as PNG for training/visual inspection.

---

This control panel enables precise editing, visualization, and extraction of map data for simulations and training pipelines.
