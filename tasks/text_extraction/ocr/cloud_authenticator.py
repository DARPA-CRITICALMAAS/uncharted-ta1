import logging, os

import msal
import requests
from google.oauth2.credentials import Credentials
from google.cloud.vision import ImageAnnotatorClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Provided USGS
class CloudAuthenticator:
    def __init__(self):
        # Validate environment variables
        self.validate_environment_variables()

        # Entra ID (Azure AD) Configuration
        self.client_id = os.environ.get("AZURE_CLIENT_ID")
        self.client_secret = os.environ.get("AZURE_CLIENT_SECRET")
        self.tenant_id = os.environ.get("AZURE_TENANT_ID")
        self.azure_audience = os.environ.get(
            "AZURE_AUDIENCE"
        )  # Should match the app registration API URI

        # Google Cloud Configuration
        self.google_audience = os.environ.get(
            "GOOGLE_AUDIENCE"
        )  # Should match the workload identity audience
        self.google_sts_endpoint = "https://sts.googleapis.com/v1/token"
        logger.debug(str(self))

    def validate_environment_variables(self):
        required_vars = [
            "AZURE_CLIENT_ID",
            "AZURE_CLIENT_SECRET",
            "AZURE_TENANT_ID",
            "GOOGLE_AUDIENCE",
            "AZURE_AUDIENCE",
        ]
        for var in required_vars:
            if not os.getenv(var):
                raise EnvironmentError(f"Environment variable {var} is not set")

    def get_entra_id_token(self):
        """
        Obtain an access token from Entra ID using MSAL without using cache.
        """
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_secret
        )
        # Use /.default scope
        scopes = [f"{self.azure_audience}/.default"]
        # Always fetch a new token (no caching)
        result = app.acquire_token_for_client(scopes=scopes)
        if "access_token" in result:
            token = result["access_token"]
            expires_in = result.get("expires_in", "Unknown")
            issued_at = result.get("ext_expires_in", "Unknown")  # Some APIs provide this value
    
            logger.debug(f"New Token Acquired: {token[-10:]}...")  # Print last 10 chars for security
            logger.debug(f"Token Expires In: {expires_in} seconds")
    
            return token
        else:
            raise ValueError(f"Token acquisition failed: {result.get('error_description')}")

    def exchange_for_google_token(self, entra_token):
        """
        Exchange Entra ID token for a Google Cloud access token
        """
        payload = {
            "audience": self.google_audience,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "subject_token": entra_token,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
        }

        response = requests.post(self.google_sts_endpoint, json=payload)
        logger.debug("Google token exchange complete")
        logger.debug("Response: " + str(response))

        if response.status_code != 200:
            raise ValueError(f"Token exchange failed: {response.text}")

        return response.json()["access_token"]

    def get_vision_client(self):
        """
        Get an authenticated Google Cloud Vision client
        """
        try:
            # Step 1: Get the Entra ID token
            entra_token = self.get_entra_id_token()
        except ValueError as e:
            raise ValueError(f"Failed to acquire Entra ID token: {e}")

        try:
            # Step 2: Exchange the Entra ID token for a Google Cloud access token
            google_access_token = self.exchange_for_google_token(entra_token)
        except ValueError as e:
            raise ValueError(
                f"Failed to exchange Entra ID token for Google Cloud token: {e}"
            )

        # Step 3: Return the Vision client with the Google token
        vision_credentials = Credentials(google_access_token)
        return ImageAnnotatorClient(credentials=vision_credentials)
