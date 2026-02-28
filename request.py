import requests
import time

BASE_URL = "https://hackathon20.arke.so/api"
USERNAME = "arke"
PASSWORD = "arke"

class ArkeAPI:
    def __init__(self):
        self.token = "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3MjFkZDc1My1kMmIzLTRkNjctODQ1NS1lNGUzZmE1YjZiNTIiLCJleHAiOjE3NzI0ODgwNTQsImlhdCI6MTc3MjIyODg1NCwiZnVsbF9uYW1lIjoiYXJrZSIsInVzZXJuYW1lIjoiYXJrZSIsInRlbmFudCI6eyJ0ZW5hbnRfaWQiOiJjMzFjZDI3Ny02NDRiLTQ1NzMtYWQ3Yi0yMzUzM2ZjODNjOTIiLCJ0ZW5hbnRfdXJsIjoiaGFja2F0aG9uMjAifSwic3VwZXJfYWRtaW4iOmZhbHNlLCJyb2xlcyI6WyJhZG1pbiJdfQ.loxWLefUyjCXjthEXxjhxt0K4KC9ULe1mXqQV75huTLJHJ5rtlSuk5HDjAZuQOVeWyKeKfkVBBxW7Ws1I4YyrBEDE7ftUO4bmnqGL_krgwIUJGb4JLB_XRzQd4H0ujC5dBQPaGWcxY_Ra07jiB9onS5oZ9OLjkF_ePq1GFH1-mZwFWV9791AK1GfUZ_qDmzGRempcxxDE4Un0wMQeF0C0xqF2VxTb5KjCffesagocbsgOhGaxT-QVytL1dR-cHa8UvNd7WKWpsOdkG2ynjlkzaVg0874FzwbNGA6PuP_eEgi8FP7Wowc65dzFyJERXcrgTG-dNZ26zDVsiHrH8BMkg"
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