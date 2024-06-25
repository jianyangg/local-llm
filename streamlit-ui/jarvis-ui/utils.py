import hashlib

def generate_hasher(unique_id):
    # Create a new sha256 hash object
    sha_signature = hashlib.sha256(unique_id.encode()).hexdigest()
    return sha_signature

def generate_tenant_id(username, password):
    return generate_hasher(username + password)