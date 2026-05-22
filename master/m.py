#!/usr/bin/env python3
import asyncio
import os
import random
from pyppeteer import launch

async def run():
    # === Configuration ===

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        raise FileNotFoundError(f"Chrome not found at: {chrome_path}")

    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    html_file = os.path.join(desktop, "pathconvergance.html")
    if not os.path.exists(html_file):
        raise FileNotFoundError(f"HTML file not found: {html_file}")
    url = f"file:///{html_file.replace(os.sep, '/')}"

    image_path = os.path.join(
        desktop,
        "overlay_t1750998942_w-116_399930_s33_715240_e-116_356460_n33_734730.png"
    )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Overlay image not found: {image_path}")

    # === Create full SegNet folder structure ===
    dataset_root = os.path.join(desktop, "signet_dataset")
    os.makedirs(dataset_root, exist_ok=True)
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(dataset_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, split, "labels"), exist_ok=True)

    # === Launch browser ===
    browser = await launch(
        headless=False,
        executablePath=chrome_path,
        defaultViewport=None,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--allow-running-insecure-content",
        ],
    )
    page = await browser.newPage()

    await page.goto(url)
    await page.waitForSelector("#mapUpload")
    up = await page.querySelector("#mapUpload")
    await up.uploadFile(image_path)
    await page.waitForSelector("#usergui1")
    await page.click("#usergui1")
    await asyncio.sleep(1)

    num_cycles = 100  # <-- Change to 200 later if needed

    for cycle in range(1, num_cycles + 1):
        print(f"--- Cycle {cycle}/{num_cycles} ---")

        # Randomly assign split per cycle
        rand_val = random.random()
        if rand_val < 0.7:
            split = "train"
        elif rand_val < 0.85:
            split = "val"
        else:
            split = "test"

        await page.evaluate("""
          () => {
            const p = document.getElementById('pause');
            const r = document.getElementById('runningBox');
            p.checked = true;  p.dispatchEvent(new Event('change'));
            r.checked = true;  r.dispatchEvent(new Event('change'));
          }
        """)
        await asyncio.sleep(3)

        await page.evaluate("""
          () => {
            const p = document.getElementById('pause');
            p.checked = false;  p.dispatchEvent(new Event('change'));
          }
        """)
        await asyncio.sleep(1)
        await page.evaluate("document.getElementById('map').scrollIntoView()")
        overlay_shot = os.path.join(dataset_root, split, "labels", f"{cycle:04d}.png")
        await page.screenshot({"path": overlay_shot})
        print(f"Saved overlay ({split}):", overlay_shot)

        await page.evaluate("""
          () => {
            const r = document.getElementById('runningBox');
            r.checked = false; r.dispatchEvent(new Event('change'));
          }
        """)
        await asyncio.sleep(1)
        await page.evaluate("document.getElementById('map').scrollIntoView()")
        map_shot = os.path.join(dataset_root, split, "images", f"{cycle:04d}.png")
        await page.screenshot({"path": map_shot})
        print(f"Saved map ({split}):", map_shot)

        await asyncio.sleep(1)

    print("All cycles complete. Press ENTER to close.")
    await asyncio.get_event_loop().run_in_executor(None, input)
    await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
