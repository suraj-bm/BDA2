import requests
import argparse

API_URL = "http://127.0.0.1:5000/predict"

def main(image_path):
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        r = requests.post(API_URL, files=files)
    try:
        print("Status:", r.status_code)
        print("Response:", r.json())
    except Exception:
        print("Non-JSON response:\n", r.text)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to image file to send to the server")
    args = p.parse_args()
    main(args.image)
