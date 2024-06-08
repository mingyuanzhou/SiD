# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


def read_huggingface_url(network_pkl,local_filename=None):
    # Import requests library to handle HTTP requests
    import requests
    # Specify the local filename where the downloaded file will be saved
    if local_filename is None:
        local_filename = 'sid_checkpoint.pkl'
    
    try:
        # Attempt to download the file from the provided URL
        response = requests.get(network_pkl)
        # Check for HTTP errors
        response.raise_for_status()

        # Save the downloaded content to the specified local file
        with open(local_filename, 'wb') as f:
            f.write(response.content)

        print(f"File successfully downloaded and saved as {local_filename}")

        # Update network_pkl to reference the local file path
        

    except requests.RequestException as e:
        # Handle exceptions that may occur during the download process
        print(f"Failed to download the file. Error: {e}")
    return local_filename