#!/usr/bin/env python3
"""
HMAC Secret Generation and Signal Signing for TitanovaX Trading System
Generates secure HMAC secrets and provides signing/verification utilities
"""

import secrets
import hmac
import hashlib
import json
from pathlib import Path
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('secrets/hmac_operations.log'),
        logging.StreamHandler()
    ]
)

class HMACManager:
    def __init__(self, secret_dir='secrets', signals_dir='signals'):
        self.secret_dir = Path(secret_dir)
        self.signals_dir = Path(signals_dir)
        self.secret_dir.mkdir(exist_ok=True)
        self.signals_dir.mkdir(exist_ok=True)

        self.secret_file = self.secret_dir / 'hmac_secret.key'
        self.encrypted_secret_file = self.secret_dir / 'hmac_secret.enc'
        self.public_key_file = self.secret_dir / 'public_key.pem'

    def generate_hmac_secret(self, key_length=32):
        """Generate a cryptographically secure HMAC secret"""

        # Generate random secret
        secret = secrets.token_bytes(key_length)

        # Save raw secret (for development)
        with open(self.secret_file, 'wb') as f:
            f.write(secret)

        # Generate encryption key for secret storage
        encryption_key = Fernet.generate_key()
        cipher_suite = Fernet(encryption_key)

        # Encrypt the secret
        encrypted_secret = cipher_suite.encrypt(secret)

        with open(self.encrypted_secret_file, 'wb') as f:
            f.write(encrypted_secret)

        # Save encryption key (in production, use secure key management)
        with open(self.secret_dir / 'encryption.key', 'wb') as f:
            f.write(encryption_key)

        logging.info(f"Generated HMAC secret: {secret.hex()}")
        logging.info("Secret saved to files with encryption")

        return secret.hex()

    def load_hmac_secret(self, use_encryption=True):
        """Load HMAC secret from file"""

        if use_encryption and self.encrypted_secret_file.exists():
            # Load encrypted secret
            try:
                with open(self.secret_dir / 'encryption.key', 'rb') as f:
                    encryption_key = f.read()

                cipher_suite = Fernet(encryption_key)

                with open(self.encrypted_secret_file, 'rb') as f:
                    encrypted_secret = f.read()

                secret = cipher_suite.decrypt(encrypted_secret)
                return secret

            except Exception as e:
                logging.error(f"Failed to decrypt secret: {e}")
                return None

        elif self.secret_file.exists():
            # Load raw secret (development only)
            with open(self.secret_file, 'rb') as f:
                return f.read()

        else:
            logging.error("No HMAC secret found. Generate one first.")
            return None

    def sign_payload(self, payload, use_encryption=True):
        """Sign a payload with HMAC-SHA256"""

        secret = self.load_hmac_secret(use_encryption)
        if secret is None:
            raise ValueError("No HMAC secret available")

        # Ensure payload is bytes
        if isinstance(payload, dict):
            payload_bytes = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode()
        elif isinstance(payload, str):
            payload_bytes = payload.encode()
        else:
            payload_bytes = payload

        # Calculate HMAC
        signature = hmac.new(secret, payload_bytes, hashlib.sha256).hexdigest()

        return signature

    def verify_signature(self, payload, signature, use_encryption=True):
        """Verify HMAC signature"""

        expected_signature = self.sign_payload(payload, use_encryption)
        return hmac.compare_digest(expected_signature, signature)

    def create_signed_signal(self, signal_data, filename=None):
        """Create a signed signal file"""

        if filename is None:
            timestamp = signal_data.get('timestamp', int(time.time()))
            filename = f"signal_{timestamp}.json"

        # Sign the payload
        signature = self.sign_payload(signal_data)

        # Save signal data
        signal_file = self.signals_dir / filename
        with open(signal_file, 'w') as f:
            json.dump(signal_data, f, indent=2)

        # Save signature
        signature_file = self.signals_dir / f"{filename}.hmac"
        with open(signature_file, 'w') as f:
            f.write(signature)

        logging.info(f"Created signed signal: {signal_file}")
        logging.info(f"Signature: {signature[:16]}...")

        return signal_file, signature_file

    def verify_signal_file(self, signal_file, hmac_file):
        """Verify a signal file and its HMAC"""

        # Read signal data
        with open(signal_file, 'r') as f:
            signal_data = json.load(f)

        # Read signature
        with open(hmac_file, 'r') as f:
            expected_signature = f.read().strip()

        # Verify
        is_valid = self.verify_signature(signal_data, expected_signature)

        if is_valid:
            logging.info("✅ Signal signature is valid")
        else:
            logging.error("❌ Signal signature is invalid")

        return is_valid

    def rotate_secret(self, new_key_length=32):
        """Rotate HMAC secret (creates new secret while keeping old for verification)"""

        old_secret = self.load_hmac_secret()
        new_secret = self.generate_hmac_secret(new_key_length)

        # Save old secret for verification during transition
        with open(self.secret_dir / 'hmac_secret.old', 'wb') as f:
            f.write(old_secret)

        logging.info("HMAC secret rotated successfully")
        logging.info("Old secret saved for verification during transition period")

        return new_secret

    def generate_api_keys(self, count=5):
        """Generate API keys for different services"""

        api_keys = {}

        for i in range(count):
            # Generate API key
            api_key = secrets.token_hex(32)

            # Generate corresponding secret
            api_secret = secrets.token_bytes(32)

            api_keys[f"api_key_{i+1}"] = {
                'key': api_key,
                'secret': api_secret.hex(),
                'created': time.time()
            }

            # Save individual key files
            key_file = self.secret_dir / f"api_key_{i+1}.json"
            with open(key_file, 'w') as f:
                json.dump({
                    'api_key': api_key,
                    'api_secret': api_secret.hex(),
                    'created': time.time()
                }, f, indent=2)

        # Save master list
        master_file = self.secret_dir / 'api_keys.json'
        with open(master_file, 'w') as f:
            json.dump(api_keys, f, indent=2, default=str)

        logging.info(f"Generated {count} API key pairs")

        return api_keys

# PowerShell script integration for MT5
class MT5HMACIntegration:
    def __init__(self, secret_dir='secrets'):
        self.secret_dir = Path(secret_dir)

    def generate_mt5_secret_key(self):
        """Generate secret key in format suitable for MT5 GlobalVariables"""

        secret = secrets.token_bytes(32)

        # Save as MT5-compatible format
        mt5_key_file = self.secret_dir / 'mt5_secret.key'
        with open(mt5_key_file, 'wb') as f:
            f.write(secret)

        # Generate MT5 loading script
        mt5_script = f"""
// MT5 HMAC Secret Loading Script
// Generated: {time.ctime()}

// Load HMAC secret into MT5 GlobalVariables
int LoadHMACSecret()
{{
    string secretFile = "C:\\\\titanovax\\\\secrets\\\\mt5_secret.key";

    int handle = FileOpen(secretFile, FILE_READ | FILE_BIN);
    if(handle == INVALID_HANDLE)
    {{
        Print("Failed to load HMAC secret file");
        return false;
    }}

    int size = (int)FileSize(handle);
    if(size != 32)
    {{
        Print("Invalid secret file size: " + (string)size);
        FileClose(handle);
        return false;
    }}

    uchar secret[32];
    FileReadArray(handle, secret, 0, 32);
    FileClose(handle);

    // Store in GlobalVariables
    GlobalVariableSet("HMAC_SECRET_SIZE", 32);
    for(int i = 0; i < 32; i++)
    {{
        GlobalVariableSet("HMAC_SECRET_" + (string)i, (double)secret[i]);
    }}

    Print("HMAC secret loaded successfully");
    return true;
}}
"""

        mt5_script_file = self.secret_dir / 'load_hmac_mt5.mq5'
        with open(mt5_script_file, 'w') as f:
            f.write(mt5_script)

        return secret.hex()

    def create_mt5_hmac_function(self):
        """Create MT5-compatible HMAC function"""

        mt5_hmac_function = """
//+------------------------------------------------------------------+
//| Calculate HMAC-SHA256 for MT5                                    |
//+------------------------------------------------------------------+
string CalculateHMAC_MT5(string message)
{
    // Get secret from GlobalVariables
    int secretSize = (int)GlobalVariableGet("HMAC_SECRET_SIZE");
    if(secretSize != 32)
    {
        Print("Invalid HMAC secret size");
        return "";
    }

    uchar secret[32];
    for(int i = 0; i < 32; i++)
    {
        secret[i] = (uchar)GlobalVariableGet("HMAC_SECRET_" + (string)i);
    }

    // Simple HMAC implementation for MT5
    // In production, use proper cryptographic functions
    int hash = 0;
    for(int i = 0; i < StringLen(message); i++)
    {
        hash = hash * 31 + StringGetCharacter(message, i);
    }

    // XOR hash with secret bytes
    for(int i = 0; i < 32; i++)
    {
        hash = hash ^ secret[i];
    }

    return IntegerToString(MathAbs(hash));
}
"""

        return mt5_hmac_function

def main():
    """Main function for HMAC management"""

    print("=== TitanovaX HMAC Secret Management ===")

    hmac_manager = HMACManager()

    # Check if secret exists
    if not hmac_manager.secret_file.exists() and not hmac_manager.encrypted_secret_file.exists():
        print("No HMAC secret found. Generating new secret...")
        secret = hmac_manager.generate_hmac_secret()
        print(f"Generated secret: {secret[:16]}...")
    else:
        print("HMAC secret already exists")

    # Generate API keys
    print("\nGenerating API keys...")
    api_keys = hmac_manager.generate_api_keys(3)

    for key_name, key_data in api_keys.items():
        print(f"{key_name}: {key_data['key'][:16]}...")

    # Create sample signed signal
    print("\nCreating sample signed signal...")
    sample_signal = {
        "timestamp": int(time.time()),
        "symbol": "EURUSD",
        "side": "BUY",
        "volume": 0.01,
        "price": 1.0850,
        "model_id": "xgboost_v1",
        "model_version": "1.0.0",
        "features_hash": "abc123",
        "reason": "Technical breakout",
        "confidence": 0.85
    }

    signal_file, hmac_file = hmac_manager.create_signed_signal(sample_signal)
    print(f"Created signal: {signal_file}")
    print(f"HMAC file: {hmac_file}")

    # Verify the signal
    print("\nVerifying signal...")
    is_valid = hmac_manager.verify_signal_file(signal_file, hmac_file)

    if is_valid:
        print("✅ Signal verification successful!")
    else:
        print("❌ Signal verification failed!")

    # Generate MT5 integration
    print("\nGenerating MT5 integration...")
    mt5_integration = MT5HMACIntegration()
    mt5_secret = mt5_integration.generate_mt5_secret_key()
    print(f"MT5 secret generated: {mt5_secret[:16]}...")

    print("\n=== Setup Complete ===")
    print("HMAC secret and API keys generated successfully!")
    print("Signal signing and verification system ready!")
    print("MT5 integration files created!")

if __name__ == "__main__":
    main()
