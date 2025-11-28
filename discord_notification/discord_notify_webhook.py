import requests
import json
import os

# Replace with your own Discord webhook URL
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_webhook_message(message, webhook_url=DISCORD_WEBHOOK_URL):
    """Send a message to a Discord channel via webhook."""
    data = {
        'content': message,  # The message content
        # 'username': 'Webhook Bot',  # Optional: Customize the name of the sender
    }

    response = requests.post(
        webhook_url, data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )

    try:
        response.raise_for_status()  # Check for HTTP request errors
    except requests.exceptions.HTTPError as err:
        print(f'Error sending message: {err}')
    else:
        print(f'Message sent successfully, status code: {response.status_code}')


if __name__ == '__main__':
    # Example: Send a message to the Discord channel
    msg = '```\nDiscord APP/Bot Test 0\ndev_bot\n```'
    send_webhook_message(msg, YOUR_DISCORD_WEBHOOK_URL)
