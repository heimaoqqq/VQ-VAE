"""
A simple script to inspect the __init__ signature of the VectorQuantizer class
from the installed diffusers library. This helps resolve parameter name mismatches
between different library versions.
"""
import inspect
from diffusers.models.autoencoders.vq_model import VectorQuantizer

try:
    print("--- Inspecting diffusers.models.autoencoders.vq_model.VectorQuantizer ---")
    signature = inspect.signature(VectorQuantizer.__init__)
    print(f"Signature: {signature}")
    print("\nParameters found:")
    for name, param in signature.parameters.items():
        print(f"- Name: {name}, Default: {param.default}")
except Exception as e:
    print(f"An error occurred during inspection: {e}")
    print("Please ensure the diffusers library is correctly installed in the environment.") 