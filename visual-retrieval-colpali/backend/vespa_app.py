import os
from vespa.application import Vespa
from dotenv import load_dotenv


def get_vespa_app():
    load_dotenv()
    vespa_app_url = os.environ.get(
        "VESPA_APP_URL"
    )  # Ensure this is set to your Vespa app URL
    vespa_cloud_secret_token = os.environ.get("VESPA_CLOUD_SECRET_TOKEN")

    if not vespa_app_url or not vespa_cloud_secret_token:
        raise ValueError(
            "Please set the VESPA_APP_URL and VESPA_CLOUD_SECRET_TOKEN environment variables"
        )
    # Instantiate Vespa connection
    vespa_app = Vespa(
        url=vespa_app_url, vespa_cloud_secret_token=vespa_cloud_secret_token
    )
    vespa_app.wait_for_application_up()
    print(f"Connected to Vespa at {vespa_app_url}")
    return vespa_app
