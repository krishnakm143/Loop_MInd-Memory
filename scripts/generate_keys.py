"""Generate Ed25519 key pair for agent signing."""

import os
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from api.config import settings


def generate_key_pair():
    """Generate Ed25519 private and public keys."""
    # Create keys directory if it doesn't exist
    keys_dir = Path(settings.ed25519_private_key_path).parent
    keys_dir.mkdir(parents=True, exist_ok=True)

    # Generate private key
    private_key = ed25519.Ed25519PrivateKey.generate()

    # Serialize private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Get public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    # Write keys to files
    private_key_path = Path(settings.ed25519_private_key_path)
    public_key_path = Path(settings.ed25519_public_key_path)

    with open(private_key_path, "wb") as f:
        f.write(private_pem)

    with open(public_key_path, "wb") as f:
        f.write(public_pem)

    # Set restrictive permissions (Unix-like systems)
    if os.name != "nt":
        os.chmod(private_key_path, 0o600)
        os.chmod(public_key_path, 0o644)

    print(f"Generated Ed25519 key pair:")
    print(f"  Private key: {private_key_path}")
    print(f"  Public key: {public_key_path}")


if __name__ == "__main__":
    generate_key_pair()

