import os

def load_api_key():
    """Load API key from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key == 'GOOGLE_API_KEY':
                        os.environ['GOOGLE_API_KEY'] = value
                        return True
    return False

if __name__ == "__main__":
    if load_api_key():
        print("API key loaded successfully!")
    else:
        print("Failed to load API key. Make sure .env file exists with GOOGLE_API_KEY=your_key")