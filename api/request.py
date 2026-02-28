import requests
import time
import os

BASE_URL = os.getenv("ARKE_BASE_URL", "https://hackathon20.arke.so/api")
USERNAME = os.getenv("ARKE_USERNAME", "")
PASSWORD = os.getenv("ARKE_PASSWORD", "arke")

class ArkeAPI:
    def __init__(self):
        self.token = os.getenv("ARKE_TOKEN", "")
        self.token_expiry = 0

    def login(self):
        url = f"{BASE_URL}/login"
        payload = {
            "username": USERNAME,
            "password": PASSWORD
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        self.token = data["accessToken"]

        # Token valid for 72 hours (259200 seconds)
        self.token_expiry = time.time() + 259200

        print(" Logged in successfully.")

    def get_headers(self):
        if not self.token or time.time() >= self.token_expiry:
            print(" Token expired or missing. Logging in again...")
            self.login()

        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def get(self, endpoint):
        url = f"{BASE_URL}{endpoint}"
        headers = self.get_headers()

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint, data=None):
        url = f"{BASE_URL}{endpoint}"
        headers = self.get_headers()

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def put(self, endpoint, data=None):
        url = f"{BASE_URL}{endpoint}"
        headers = self.get_headers()

        response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()


# =====================
# Usage Example
# =====================

if __name__ == "__main__":
    api = ArkeAPI()

    # Example call (replace with real endpoint)
    try:
        result = api.get("/sales/order")
        print("API Response:")
        print(result)
    except requests.exceptions.HTTPError as e:
        print("API Error:", e)